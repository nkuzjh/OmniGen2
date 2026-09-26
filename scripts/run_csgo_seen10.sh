#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL="${OMNIGEN2_MODEL_PATH:-OmniGen2/OmniGen2}"
VAE_MODEL="${OMNIGEN2_VAE_MODEL_PATH:-black-forest-labs/FLUX.1-dev}"
TEXT_ENCODER_MODEL="${OMNIGEN2_TEXT_ENCODER_MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/csgo_benchmark_v2_seen10/OmniGen2"
CONFIG_PATH="${PROJECT_ROOT}/options/csgo_seen10_lora.yml"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-16}"
INFERENCE_DECODE_BATCH_SIZE="${INFERENCE_DECODE_BATCH_SIZE:-1}"

# Xet token refreshes are unreliable through the proxy used on this server.
# The regular Hub HTTP downloader supports resumable downloads and avoids the
# failing /xet-read-token request. Set HF_HUB_DISABLE_XET=0 explicitly to opt in.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"

usage() {
    echo "Usage: $0 {smoke|train|convert|infer|eval|all} [--seed N] [--task discrete|continuous|all] [--resume-from-checkpoint PATH|latest]"
    echo "       paths: [--data-root PATH] [--eval-root PATH] [--eval-python PATH] [--print-paths]"
    echo "       infer options: [--batch-size N] [--vae-decode-batch-size N] [--no-oom-fallback] [--no-fuse-lora]"
    echo "       batch defaults: INFERENCE_BATCH_SIZE=16, INFERENCE_DECODE_BATCH_SIZE=1"
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

MODEL_PYTHON_CANDIDATE="${OMNIGEN2_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
if [[ "$MODEL_PYTHON_CANDIDATE" == "~/"* ]]; then
    MODEL_PYTHON_CANDIDATE="${HOME}/${MODEL_PYTHON_CANDIDATE:2}"
elif [[ "$MODEL_PYTHON_CANDIDATE" != */* ]] && command -v "$MODEL_PYTHON_CANDIDATE" >/dev/null 2>&1; then
    MODEL_PYTHON_CANDIDATE="$(command -v "$MODEL_PYTHON_CANDIDATE")"
elif [[ "$MODEL_PYTHON_CANDIDATE" != /* ]]; then
    MODEL_PYTHON_CANDIDATE="${PROJECT_ROOT}/${MODEL_PYTHON_CANDIDATE}"
fi
if [[ -x "$MODEL_PYTHON_CANDIDATE" ]]; then
    RESOLVER_PYTHON="$MODEL_PYTHON_CANDIDATE"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    RESOLVER_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    RESOLVER_PYTHON="$(command -v python3)"
else
    echo "Python 3 is required for CSGO path inspection: set OMNIGEN2_PYTHON or prepare .venv" >&2
    exit 2
fi

# The aligned profile has a separate parser/run root. Without --experiment the
# original commands, seed defaults and checkpoint behavior remain unchanged.
for argument in "$@"; do
    if [[ "$argument" == "--experiment" || "$argument" == --experiment=* ]]; then
        exec "$RESOLVER_PYTHON" "$PROJECT_ROOT/scripts/run_csgo_aligned.py" "$@"
    fi
done

ACTION="$1"
shift
SEED=0
TASK=all
RESUME_FROM_CHECKPOINT=""
INFER_EXTRA_ARGS=()
DATA_ROOT_ARG=""
EVAL_ROOT_ARG=""
EVAL_PYTHON_ARG=""
HAS_DATA_ROOT=0
HAS_EVAL_ROOT=0
HAS_EVAL_PYTHON=0
PRINT_PATHS=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root|--eval-root|--eval-python|--unilip-python)
            [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
            case "$1" in
                --data-root) DATA_ROOT_ARG="$2"; HAS_DATA_ROOT=1 ;;
                --eval-root) EVAL_ROOT_ARG="$2"; HAS_EVAL_ROOT=1 ;;
                *) EVAL_PYTHON_ARG="$2"; HAS_EVAL_PYTHON=1 ;;
            esac
            shift 2
            ;;
        --data-root=*|--eval-root=*|--eval-python=*|--unilip-python=*)
            case "$1" in
                --data-root=*) DATA_ROOT_ARG="${1#*=}"; HAS_DATA_ROOT=1 ;;
                --eval-root=*) EVAL_ROOT_ARG="${1#*=}"; HAS_EVAL_ROOT=1 ;;
                *) EVAL_PYTHON_ARG="${1#*=}"; HAS_EVAL_PYTHON=1 ;;
            esac
            shift
            ;;
        --print-paths)
            PRINT_PATHS=1
            shift
            ;;
        --seed)
            [[ $# -ge 2 ]] || { echo "--seed requires a value" >&2; exit 2; }
            SEED="$2"
            shift 2
            ;;
        --seed=*)
            SEED="${1#*=}"
            shift
            ;;
        --task)
            [[ $# -ge 2 ]] || { echo "--task requires a value" >&2; exit 2; }
            TASK="$2"
            shift 2
            ;;
        --task=*)
            TASK="${1#*=}"
            shift
            ;;
        --resume-from-checkpoint)
            [[ $# -ge 2 ]] || { echo "--resume-from-checkpoint requires a value" >&2; exit 2; }
            RESUME_FROM_CHECKPOINT="$2"
            shift 2
            ;;
        --resume-from-checkpoint=*)
            RESUME_FROM_CHECKPOINT="${1#*=}"
            shift
            ;;
        --batch-size)
            [[ "$ACTION" == "infer" || "$ACTION" == "all" ]] || {
                echo "--batch-size is only valid for infer or all" >&2
                exit 2
            }
            [[ $# -ge 2 ]] || { echo "--batch-size requires a value" >&2; exit 2; }
            INFERENCE_BATCH_SIZE="$2"
            shift 2
            ;;
        --batch-size=*)
            [[ "$ACTION" == "infer" || "$ACTION" == "all" ]] || {
                echo "--batch-size is only valid for infer or all" >&2
                exit 2
            }
            INFERENCE_BATCH_SIZE="${1#*=}"
            shift
            ;;
        --vae-decode-batch-size)
            [[ "$ACTION" == "infer" || "$ACTION" == "all" ]] || {
                echo "--vae-decode-batch-size is only valid for infer or all" >&2
                exit 2
            }
            [[ $# -ge 2 ]] || { echo "--vae-decode-batch-size requires a value" >&2; exit 2; }
            INFERENCE_DECODE_BATCH_SIZE="$2"
            shift 2
            ;;
        --vae-decode-batch-size=*)
            [[ "$ACTION" == "infer" || "$ACTION" == "all" ]] || {
                echo "--vae-decode-batch-size is only valid for infer or all" >&2
                exit 2
            }
            INFERENCE_DECODE_BATCH_SIZE="${1#*=}"
            shift
            ;;
        --no-oom-fallback|--no-fuse-lora)
            [[ "$ACTION" == "infer" || "$ACTION" == "all" ]] || {
                echo "$1 is only valid for infer or all" >&2
                exit 2
            }
            INFER_EXTRA_ARGS+=("$1")
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

PATH_ARGS=(--action "$ACTION")
(( HAS_DATA_ROOT == 0 )) || PATH_ARGS+=(--data-root "$DATA_ROOT_ARG")
(( HAS_EVAL_ROOT == 0 )) || PATH_ARGS+=(--eval-root "$EVAL_ROOT_ARG")
(( HAS_EVAL_PYTHON == 0 )) || PATH_ARGS+=(--eval-python "$EVAL_PYTHON_ARG")
if (( PRINT_PATHS )); then
    exec "$RESOLVER_PYTHON" "$PROJECT_ROOT/csgo_runtime_paths.py" "${PATH_ARGS[@]}"
fi
PATH_LINES="$("$RESOLVER_PYTHON" "$PROJECT_ROOT/csgo_runtime_paths.py" "${PATH_ARGS[@]}" --lines)"
mapfile -t RESOLVED_PATHS <<< "$PATH_LINES"
PROJECT_PYTHON="${RESOLVED_PATHS[0]}"
DATA_ROOT="${RESOLVED_PATHS[1]}"
SHARED_EVAL_DIR="${RESOLVED_PATHS[2]}"
UNILIP_PYTHON="${RESOLVED_PATHS[3]}"

if [[ "$ACTION" == "infer" || "$ACTION" == "all" ]]; then
    [[ "$INFERENCE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
        echo "INFERENCE_BATCH_SIZE / --batch-size must be a positive integer" >&2
        exit 2
    }
    [[ "$INFERENCE_DECODE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
        echo "INFERENCE_DECODE_BATCH_SIZE / --vae-decode-batch-size must be a positive integer" >&2
        exit 2
    }
fi

[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "--seed must be a non-negative integer" >&2; exit 2; }
case "$TASK" in
    discrete|continuous|all) ;;
    *) echo "--task must be discrete, continuous, or all" >&2; exit 2 ;;
esac
[[ -x "$PROJECT_PYTHON" ]] || { echo "Project Python is not executable: $PROJECT_PYTHON" >&2; exit 2; }
if [[ "$ACTION" == "train" || "$ACTION" == "infer" || "$ACTION" == "all" || "$ACTION" == "smoke" || "$ACTION" == "eval" ]]; then
    [[ -d "$DATA_ROOT" ]] || { echo "Data root is not a directory: $DATA_ROOT" >&2; exit 2; }
fi
if [[ "$ACTION" == "eval" || "$ACTION" == "all" || "$ACTION" == "smoke" ]]; then
    [[ -f "${SHARED_EVAL_DIR}/run_eval.py" ]] || { echo "Evaluator is missing: ${SHARED_EVAL_DIR}/run_eval.py" >&2; exit 2; }
    [[ -x "$UNILIP_PYTHON" ]] || { echo "Evaluator Python is not executable: $UNILIP_PYTHON" >&2; exit 2; }
fi

SEED_ROOT="${OUTPUT_ROOT}/seed_${SEED}"
TRAIN_ROOT="${SEED_ROOT}/train"
ADAPTER_ROOT="${TRAIN_ROOT}/inference_adapter_best"

run_smoke() {
    cd "$PROJECT_ROOT"
    "$PROJECT_PYTHON" -m pytest -q \
        tests/test_csgo_seen10_dataset.py \
        tests/test_pose_conditioning.py \
        tests/test_csgo_training_helpers.py \
        tests/test_infer_seen10_batching.py \
        tests/test_pipeline_inference_caches.py
    if [[ ! -e "${SEED_ROOT}/smoke" ]]; then
        "$PROJECT_PYTHON" smoke_seen10.py \
            --data-root "$DATA_ROOT" \
            --output-root "$OUTPUT_ROOT" \
            --seed "$SEED"
    else
        echo "Smoke artifacts already exist and will not be overwritten: ${SEED_ROOT}/smoke"
    fi
    "$UNILIP_PYTHON" "${SHARED_EVAL_DIR}/run_eval.py" smoke discrete \
        --pred-root "${SEED_ROOT}/smoke/discrete" \
        --data-root "$DATA_ROOT" \
        --limit 1 \
        --device cpu
}

run_train() {
    cd "$PROJECT_ROOT"
    local num_processes="${NUM_PROCESSES:-1}"
    [[ "$num_processes" =~ ^[1-9][0-9]*$ ]] || {
        echo "NUM_PROCESSES must be a positive integer" >&2
        exit 2
    }
    local launch_args=(
        --num_machines 1
        --num_processes "$num_processes"
        --mixed_precision bf16
    )
    if (( num_processes > 1 )); then
        launch_args+=(
            --use_fsdp
            --fsdp_offload_params false
            --fsdp_sharding_strategy HYBRID_SHARD_ZERO2
            --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP
            --fsdp_transformer_layer_cls_to_wrap OmniGen2TransformerBlock
            --fsdp_state_dict_type FULL_STATE_DICT
            --fsdp_forward_prefetch false
            --fsdp_use_orig_params true
            --fsdp_cpu_ram_efficient_loading false
            --fsdp_sync_module_states true
        )
    fi
    local train_args=(
        train_seen10.py
        --config "$CONFIG_PATH"
        --seed "$SEED"
        --output-root "$OUTPUT_ROOT"
        --data-root "$DATA_ROOT"
        --pretrained-model-path "$BASE_MODEL"
        --pretrained-vae-model-path "$VAE_MODEL"
        --pretrained-text-encoder-model-path "$TEXT_ENCODER_MODEL"
    )
    if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
        train_args+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
    fi
    "$PROJECT_PYTHON" -m accelerate.commands.launch "${launch_args[@]}" "${train_args[@]}"
}

run_convert() {
    cd "$PROJECT_ROOT"
    [[ -d "${TRAIN_ROOT}/best" ]] || {
        echo "Best checkpoint is missing: ${TRAIN_ROOT}/best" >&2
        exit 2
    }
    "$PROJECT_PYTHON" convert_ckpt_to_hf_format.py \
        --config_path "$CONFIG_PATH" \
        --model_path "${TRAIN_ROOT}/best" \
        --save_path "$ADAPTER_ROOT"
}

run_infer() {
    cd "$PROJECT_ROOT"
    [[ -d "$ADAPTER_ROOT" ]] || {
        echo "Converted best adapter is missing: $ADAPTER_ROOT (run convert first)" >&2
        exit 2
    }
    "$PROJECT_PYTHON" infer_seen10.py \
        --task "$TASK" \
        --seed "$SEED" \
        --data-root "$DATA_ROOT" \
        --output-root "$SEED_ROOT" \
        --model-path "$BASE_MODEL" \
        --adapter-path "$ADAPTER_ROOT" \
        --num-inference-steps "${NUM_INFERENCE_STEPS:-28}" \
        --dtype "${INFERENCE_DTYPE:-bf16}" \
        --batch-size "$INFERENCE_BATCH_SIZE" \
        --vae-decode-batch-size "$INFERENCE_DECODE_BATCH_SIZE" \
        "${INFER_EXTRA_ARGS[@]}"
}

eval_one() {
    local eval_task="$1"
    "$UNILIP_PYTHON" "${SHARED_EVAL_DIR}/run_eval.py" "$eval_task" \
        --pred-root "${SEED_ROOT}/${eval_task}" \
        --data-root "$DATA_ROOT" \
        --output "${SEED_ROOT}/evaluation/${eval_task}"
}

run_eval() {
    if [[ "$TASK" == "all" || "$TASK" == "discrete" ]]; then
        eval_one discrete
    fi
    if [[ "$TASK" == "all" || "$TASK" == "continuous" ]]; then
        eval_one continuous
    fi
}

case "$ACTION" in
    smoke) run_smoke ;;
    train) run_train ;;
    convert) run_convert ;;
    infer) run_infer ;;
    eval) run_eval ;;
    all)
        run_train
        run_convert
        run_infer
        run_eval
        ;;
    *)
        echo "Unknown action: $ACTION" >&2
        usage >&2
        exit 2
        ;;
esac
