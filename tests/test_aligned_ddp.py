"""Real two-process CPU DDP check of the aligned accumulation normalization."""
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(73)
        base = torch.nn.Linear(3, 2, dtype=torch.float64)
        reference = torch.nn.Linear(3, 2, dtype=torch.float64)
        reference.load_state_dict(base.state_dict())
        generator = torch.Generator().manual_seed(42)
        x = torch.randn(128, 3, generator=generator, dtype=torch.float64)
        target = torch.randn(128, 2, generator=generator, dtype=torch.float64)
        reference_loss = (reference(x) - target).square().mean()
        reference_loss.backward()
        model = DistributedDataParallel(base, find_unused_parameters=True)
        micro, accumulation = 4, 16  # 2 * 4 * 16 == 128
        for index in range(accumulation):
            start = rank * 64 + index * micro
            with model.no_sync() if index < accumulation - 1 else nullcontext():
                numerator = (model(x[start:start+micro]) - target[start:start+micro]).square().sum()
                # Same global token normalization as production, with 2 targets/sample.
                (numerator * 2 / (128 * 2)).backward()
        for actual, expected in zip(base.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-12, atol=1e-12)
        optimizer = torch.optim.AdamW(base.parameters(), lr=8e-7, betas=(.9, .95), weight_decay=.01)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=8e-7, betas=(.9, .95), weight_decay=.01)
        optimizer.step()
        reference_optimizer.step()
        for actual, expected in zip(base.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    finally:
        dist.destroy_process_group()


def test_two_process_ddp_accumulation_matches_effective_batch_128(tmp_path):
    spawn(_worker, args=(str(Path(tmp_path) / "gloo-rendezvous"),), nprocs=2, join=True)


class _OptionalReference(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Linear(3, 2, dtype=torch.float64)
        self.reference = torch.nn.Linear(3, 2, dtype=torch.float64)

    def forward(self, x, use_reference):
        result = self.main(x)
        if use_reference.any():
            result = result + self.reference(x) * use_reference[:, None]
        return result


def _optional_branch_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(73)
        base = _OptionalReference()
        reference = _OptionalReference()
        reference.load_state_dict(base.state_dict())
        generator = torch.Generator().manual_seed(42)
        x = torch.randn(128, 3, generator=generator, dtype=torch.float64)
        target = torch.randn(128, 2, generator=generator, dtype=torch.float64)
        use_reference = torch.ones(128, dtype=torch.float64)
        # Rank zero's final microbatch skips the branch. Earlier micros use it.
        use_reference[60:64] = 0
        result = reference.main(x) + reference.reference(x) * use_reference[:, None]
        (result - target).square().mean().backward()
        model = DistributedDataParallel(base, find_unused_parameters=True)
        for index in range(16):
            start = rank * 64 + index * 4
            with model.no_sync() if index < 15 else nullcontext():
                prediction = model(x[start:start+4], use_reference[start:start+4])
                numerator = (prediction - target[start:start+4]).square().sum()
                (numerator * 2 / (128 * 2)).backward()
        for actual, expected in zip(base.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-12, atol=1e-12)
    finally:
        dist.destroy_process_group()


def test_ddp_accumulation_when_last_microbatch_skips_reference_branch(tmp_path):
    spawn(_optional_branch_worker, args=(str(Path(tmp_path) / "branch-rendezvous"),),
          nprocs=2, join=True)
