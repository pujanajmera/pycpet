"""
Time topology streamlines for different OpenMP thread and worker counts, and check that
the results don't depend on either.

Run it on a compute node, once for each build you want to compare (for example the
v0.1.2 release and a branch with a C change), and compare the two tables:

    python benchmarks/bench_topo_threads.py
    python benchmarks/bench_topo_threads.py --threads 1 8 32 --workers 1 8 --n-samples 1000

Each (OMP_NUM_THREADS, workers) pair runs in its own process, because the C extension
reads OMP_NUM_THREADS once, when it is loaded. The default system is the test structure
in tests/test_files; --pdb needs a structure with the same residues, since the center
and axes are chosen by atom names.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

OPTIONS = {
    "CPET_method": "topo",
    "center": {"method": "first", "atoms": [{"CD": 2}]},
    "x": {"method": "mean", "atoms": [{"CG": 1}, {"CB": 1}]},
    "y": {"method": "mean", "atoms": [{"CA": 3}, {"CB": 3}]},
    "filter": {"resid": ["GLN"], "resnum": [2]},
    "filter_intersect": True,
    "dimensions": [1.0, 1.0, 1.0],
    "step_size": 0.05,
    "initializer": "uniform",
    "max_streamline_init": "fixed_rand",
    "dtype": "float32",
}


def worker(args):
    """Compute the topology a few times in this process and report the best time."""
    from CPET.source.calculator import calculator

    options = dict(
        OPTIONS,
        n_samples=args.n_samples,
        filter_radius=args.filter_radius,
        concur_slip=args.slip,
    )
    topo = calculator(options, path_to_pdb=args.pdb)
    times = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        hist = topo.compute_topo_complete_c_shared()
        times.append(time.perf_counter() - start)
    np.save(args.output, hist)
    print(json.dumps({"best": min(times), "charges": len(topo.x)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pdb", default=str(REPO / "tests/test_files/test_large.pdb"))
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--filter-radius", type=float, default=30.0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--slip", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.output:
        return worker(args)

    import CPET

    print(f"CPET from {Path(CPET.__file__).parent}, {os.cpu_count()} CPUs visible")
    print(f"{'threads':>8} {'workers':>8} {'best time (s)':>14} {'charges':>8} {'max diff':>10}")
    reference = None
    with tempfile.TemporaryDirectory() as tmp:
        for n_threads in args.threads:
            for n_workers in args.workers:
                output = Path(tmp) / f"{n_threads}_{n_workers}.npy"
                command = [sys.executable, __file__, "--output", str(output)]
                command += ["--slip", str(n_workers), "--pdb", args.pdb]
                command += ["--n-samples", str(args.n_samples)]
                command += ["--filter-radius", str(args.filter_radius)]
                command += ["--repeat", str(args.repeat)]
                env = dict(os.environ, OMP_NUM_THREADS=str(n_threads))
                run = subprocess.run(command, env=env, capture_output=True, text=True)
                if run.returncode != 0:
                    sys.exit(run.stderr)
                report = json.loads(run.stdout.strip().splitlines()[-1])
                hist = np.load(output)
                reference = hist if reference is None else reference
                diff = np.abs(hist - reference).max()
                print(
                    f"{n_threads:>8} {n_workers:>8} {report['best']:>14.3f} "
                    f"{report['charges']:>8} {diff:>10.2g}"
                )
    print("max diff is against the first row; it should stay near zero")


if __name__ == "__main__":
    main()
