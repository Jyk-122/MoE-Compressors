"""One CPU statistics store shared by sample-parallel workers on a single host."""
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile

import torch
import torch.distributed as dist


def statistics_size(experts, hidden):
    return hidden * experts * (experts + 1) + 3  # G, q, energy/error/token totals.


@contextmanager
def shared_statistics(shapes, directory=None):
    """Create zero-filled memory-mapped storage, then remove it after all workers finish."""
    temporary = None
    location = [None]
    if dist.get_rank() == 0:
        root = Path(directory or ("/dev/shm" if os.name == "posix" else tempfile.gettempdir()))
        required = sum(statistics_size(*shape) * 8 for shape in shapes.values())
        if shutil.disk_usage(root).free < required:
            raise ValueError(f"Shared statistics need {required / 1024**3:.2f} GiB in {root}; "
                             "use --shared-stats-dir to choose a larger memory-backed directory")
        temporary = tempfile.TemporaryDirectory(prefix="optimal-scale-", dir=root)
        location[0] = temporary.name
        for target, shape in shapes.items():
            with (Path(temporary.name) / f"{target}.bin").open("wb") as file:
                file.truncate(statistics_size(*shape) * 8)
            (Path(temporary.name) / f"{target}.lock").write_bytes(b"0")
    dist.broadcast_object_list(location)
    try:
        yield Path(location[0])
        dist.barrier()
    finally:
        if temporary is not None:
            temporary.cleanup()


def map_statistics(directory, target, experts, hidden):
    storage = torch.from_file(str(Path(directory) / f"{target}.bin"), shared=True,
                              size=statistics_size(experts, hidden), dtype=torch.float64)
    gram_size = hidden * experts * experts
    gram = storage[:gram_size].view(hidden, experts, experts)
    rhs = storage[gram_size:-3].view(hidden, experts)
    return gram, rhs, storage[-3:]


@contextmanager
def statistics_lock(path):
    """Serialize additions to one layer; other layers and GPU forwards run independently."""
    if path is None:
        yield
        return
    with Path(path).open("r+b") as file:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)
