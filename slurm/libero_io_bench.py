#!/usr/bin/env python
"""I/O benchmark for the LIBERO shards. Read-only on the shards, CPU only, no GPU.

  python -u libero_io_bench.py <shard dir> [<local dir to test a copy on>]

Measures, on the node it runs on: the layout of one shard, sequential read speed (cold and cached),
random one-scene reads (cold and cached) split into read / JPEG decode / resize, how the loading speed
scales with 1, 2, 4 and 8 processes (what DataLoader workers do), the same with the data in the page
cache (the ceiling), and a copy of one shard to a local disk and random reads from that copy.
Everything is printed; if W&B is installed the run also goes to the smoke project, so the log can be read there.
"""
import os, sys, time, glob, random, subprocess, shutil, statistics, socket, re
import multiprocessing as mp
import numpy as np
import h5py
try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass
import cv2
import torch
import torch.nn.functional as F

SHARDS = sys.argv[1]
LOCAL = sys.argv[2] if len(sys.argv) > 2 else None
N_VIEWS = 7          # views read per sample (wrist camera excluded)
SIZE = 128           # the autoencoder works at 128 px
SUMMARY = {}


def log(*a):
    print(*a, flush=True)


def now():
    return time.perf_counter()


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def stats(xs):
    return f"median {1000 * statistics.median(xs):.1f} ms | p90 {1000 * pct(xs, 90):.1f} ms | max {1000 * max(xs):.1f} ms"


def drop_cache(path):
    """Ask the kernel to forget the cached pages of one file, so the next read is cold again."""
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except Exception as e:
        log("  (could not drop the cache of", os.path.basename(path), ":", e, ")")


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def dd_speed(path):
    out = sh(f"dd if={path} of=/dev/null bs=16M 2>&1 | tail -1")
    m = re.search(r"([\d.,]+)\s*([MG])B/s", out)
    if not m:
        log("  dd output:", out)
        return None
    v = float(m.group(1).replace(",", "."))
    return v * (1000 if m.group(2) == "G" else 1)


def open_shard(path):
    return h5py.File(path, "r", rdcc_nbytes=0)


def read_scene(f, off, r):
    """What Dataset.read_views does for one scene: ONE contiguous byte slice, then decode the views."""
    blob = f["data"][off[r, 0]:off[r, -1]]
    return blob


def decode_resize(blob, offr):
    out = []
    for c in range(N_VIEWS):
        lo, hi = offr[c] - offr[0], offr[c + 1] - offr[0]
        a = cv2.imdecode(blob[lo:hi], cv2.IMREAD_COLOR)
        out.append(a[:, :, ::-1].transpose(2, 0, 1))
    raw = np.ascontiguousarray(np.stack(out))
    img = torch.from_numpy(raw).float() / 255.0
    if img.shape[-1] != SIZE:
        img = F.interpolate(img, size=(SIZE, SIZE), mode="bilinear", align_corners=False, antialias=True)
    return img


def single_process(path, rows, label, drop=False):
    if drop:
        drop_cache(path)
    f = open_shard(path)
    off = f["offsets"][()].astype(np.int64)
    tr, td, nb = [], [], []
    for r in rows:
        t0 = now()
        blob = read_scene(f, off, r)
        t1 = now()
        decode_resize(blob, off[r])
        t2 = now()
        tr.append(t1 - t0)
        td.append(t2 - t1)
        nb.append(len(blob))
    f.close()
    tot = [a + b for a, b in zip(tr, td)]
    log(f"  {label}: read {stats(tr)}")
    log(f"  {label}: decode + resize {stats(td)}   (scene size {np.mean(nb) / 1024:.0f} KB)")
    log(f"  {label}: per sample {1000 * statistics.median(tot):.1f} ms median -> {1 / statistics.mean(tot):.1f} samples/s from one process")
    return statistics.median(tr), statistics.median(td), 1 / statistics.mean(tot)


def _worker(args):
    paths, n, seed = args
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    rng = random.Random(seed)
    files, offs = {}, {}
    t0 = now()
    for _ in range(n):
        si = rng.randrange(len(paths))
        if si not in files:
            files[si] = open_shard(paths[si])
            offs[si] = files[si]["offsets"][()].astype(np.int64)
        r = rng.randrange(len(offs[si]))
        blob = read_scene(files[si], offs[si], r)
        decode_resize(blob, offs[si][r])
    dt = now() - t0
    for f in files.values():
        f.close()
    return dt


def parallel(paths, P, n, seed, label):
    ctx = mp.get_context("fork")
    t0 = now()
    with ctx.Pool(P) as pool:
        times = pool.map(_worker, [(paths, n, seed * 100 + i) for i in range(P)])
    wall = now() - t0
    sps = P * n / max(max(times), 1e-9)
    log(f"  {label}: {P} processes x {n} samples -> {sps:.1f} samples/s in total ({sps / P:.1f} per process)")
    return sps


def main():
    shards = sorted(glob.glob(os.path.join(SHARDS, "shard_*.h5")))
    log("=== 0. where this runs")
    log("host", socket.gethostname(), "| cpus usable", len(os.sched_getaffinity(0)), "| torch", torch.__version__, "| cv2", cv2.__version__)
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory.current"]:
        if os.path.exists(p):
            log("cgroup", p, "=", open(p).read().strip())
    log(sh("free -g | head -2"))
    log("TMPDIR =", os.environ.get("TMPDIR"), "| SLURM_TMPDIR =", os.environ.get("SLURM_TMPDIR"))
    log(sh(f"df -hT {SHARDS} /tmp /dev/shm {LOCAL or '/tmp'} {os.path.expanduser('~')} 2>&1"))
    log(sh("ls -d /scratch* /local* /var/scratch* 2>/dev/null"))
    sizes = [os.path.getsize(p) for p in shards]
    log(f"{len(shards)} shards, total {sum(sizes) / 1e9:.1f} GB, each {min(sizes) / 1e9:.2f} to {max(sizes) / 1e9:.2f} GB")

    log("\n=== 1. layout of shard 0")
    with open_shard(shards[0]) as f:
        log("attrs:", {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()})
        for k, d in f.items():
            if isinstance(d, h5py.Dataset):
                log(f"  {k}: shape {d.shape} dtype {d.dtype} chunks {d.chunks} compression {d.compression}")
        if "data" not in f or "offsets" not in f:
            log("This benchmark needs jpeg/png shards (datasets 'data' and 'offsets'); stopping.")
            return

    # shards used for the single-shard tests: A for random reads, B for sequential reads and the cached ceiling
    A, B = shards[0], shards[-1]
    rng = random.Random(1)
    with open_shard(A) as f:
        mA = len(f["offsets"])
    rowsA = rng.sample(range(mA), min(80, mA))

    log("\n=== 2. sequential read of one whole shard (what warming the file cache would cost)")
    drop_cache(B)
    cold = dd_speed(B)
    warm = dd_speed(B)
    log(f"  cold {cold} MB/s | second read (page cache) {warm} MB/s | whole dataset cold: about {sum(sizes) / 1e6 / cold / 60:.0f} min" if cold else "  (no dd result)")
    SUMMARY.update(seq_cold_MBps=cold, seq_warm_MBps=warm)

    log("\n=== 3. random reads of single scenes from one shard, one process")
    rc, dc, sc = single_process(A, rowsA, "cold  ", drop=True)
    rw, dw, sw = single_process(A, rowsA, "cached", drop=False)
    SUMMARY.update(rand_cold_read_ms=1000 * rc, rand_cached_read_ms=1000 * rw, decode_resize_ms=1000 * dw,
                   one_process_cold_sps=sc, one_process_cached_sps=sw)

    log("\n=== 4. random scenes from ALL shards, several processes (what DataLoader workers do)")
    for P in (1, 2, 4, 8):
        n = 30 if P <= 2 else 25
        SUMMARY[f"par_cold_P{P}"] = parallel(shards, P, n, seed=P, label="cold  ")
    log("  ceiling with the data in the page cache: shard B was just read in full")
    SUMMARY["par_cached_P8"] = parallel([B], 8, 60, seed=99, label="cached")

    log("\n=== 5. copy of one shard to a local disk, then random reads there")
    cands = [c for c in [LOCAL, os.environ.get("SLURM_TMPDIR"), os.environ.get("TMPDIR"), "/tmp"] if c]
    done = False
    for c in cands:
        try:
            os.makedirs(c, exist_ok=True)
            free = shutil.disk_usage(c).free
        except Exception as e:
            log("  ", c, "not usable:", e)
            continue
        size = os.path.getsize(B)
        if free < 3 * size:
            log(f"  {c}: only {free / 1e9:.1f} GB free, skipping")
            continue
        dst = os.path.join(c, f"libero_io_bench_{os.getpid()}.h5")
        try:
            drop_cache(B)
            t0 = now()
            shutil.copyfile(B, dst)
            dt = now() - t0
            log(f"  copied {size / 1e9:.2f} GB to {c} in {dt:.0f} s ({size / 1e6 / dt:.0f} MB/s) -> whole dataset about {sum(sizes) / 1e6 / (size / 1e6 / dt) / 60:.0f} min")
            drop_cache(dst)
            with open_shard(dst) as f:
                m = len(f["offsets"])
            rows = random.Random(2).sample(range(m), min(80, m))
            lr, ld, ls = single_process(dst, rows, "local cold", drop=True)
            SUMMARY.update(local_copy_MBps=size / 1e6 / dt, local_rand_cold_ms=1000 * lr, local_one_process_sps=ls)
            done = True
        finally:
            if os.path.exists(dst):
                os.remove(dst)
        break
    if not done:
        log("  no local directory with enough free space was found")

    log("\n=== 6. what this means for training (batch 16, 64,000 steps), from the numbers above")
    for name, key in [("random reads from the network disk, 8 workers, cold", "par_cold_P8"),
                      ("8 workers with the data in the page cache (ceiling)", "par_cached_P8")]:
        sps = SUMMARY.get(key)
        if sps:
            steps = sps / 16
            log(f"  {name}: {sps:.0f} samples/s = {steps:.1f} steps/s -> {64000 / steps / 3600:.1f} h of data loading for 64k steps")
    log("done")


if __name__ == "__main__":
    run = None
    try:
        import wandb
        run = wandb.init(project="Platonic-NVS-FM-smoke", entity="mina_ai-university-of-amsterdam", name="io-bench-" + socket.gethostname())
    except Exception as e:
        print("(no W&B:", repr(e), ")", flush=True)
    try:
        main()
    finally:
        if run is not None:
            clean = {k: v for k, v in SUMMARY.items() if v is not None}
            run.summary.update(clean)
            run.finish()
