"""
Regenerate the PROCESS 5 comparison files in tests/Results/v5 with PROCESS for R.

Usage:
    python tests/fixtures/regenerate.py [--models 4 6 7] [--outcomes ols logit] [--force]
    python tests/fixtures/regenerate.py --suite mc [--cases m4_mcx1 m7_mcw1] [--force]

The default suite is the numbered models on the 2.16 test data (regenerate.R); `--suite mc` is the
multicategorical cases on tests/Data/Data_MC.csv (regenerate_mc.R, issue #17).

Requirements:
    * R, with Rscript on the PATH or given by the RSCRIPT environment variable; on Windows the newest
      installation under "C:/Program Files/R" is found automatically.
    * Hayes's PROCESS for R, a single file process.R downloaded from https://www.afhayes.com/download.html,
      given by the PROCESS_R environment variable. Its licence allows use but not redistribution, so the
      file is never part of this repository.

Existing files are skipped unless --force is given, so an interrupted run can be resumed.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "tests", "fixtures", "regenerate.R")
SCRIPT_MC = os.path.join(ROOT, "tests", "fixtures", "regenerate_mc.R")
OUT_DIR = os.path.join(ROOT, "tests", "Results", "v5")
OUT_DIR_MC = os.path.join(OUT_DIR, "mc")
ALL_MODELS = list(range(1, 77))


def find_rscript():
    candidate = os.environ.get("RSCRIPT") or shutil.which("Rscript")
    if candidate:
        return candidate
    installs = sorted(glob.glob("C:/Program Files/R/R-*/bin/Rscript.exe"))
    if installs:
        return installs[-1]
    sys.exit("Rscript not found: put it on the PATH or set RSCRIPT.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=int, nargs="+", default=ALL_MODELS, help="model numbers (default: 1 to 76)")
    parser.add_argument("--outcomes", nargs="+", choices=["ols", "logit"], default=["ols", "logit"])
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument("--suite", choices=["models", "mc"], default="models",
                        help="numbered models (default) or the multicategorical cases")
    parser.add_argument("--cases", nargs="+", default=["all"], help="case ids for --suite mc (default: all)")
    args = parser.parse_args(argv)

    process_r = os.environ.get("PROCESS_R")
    if not process_r or not os.path.exists(process_r):
        sys.exit("Set PROCESS_R to the path of Hayes's process.R (download it from https://www.afhayes.com/download.html).")

    if args.suite == "mc":
        command = [find_rscript(), SCRIPT_MC, process_r, ROOT, OUT_DIR_MC, ",".join(args.cases),
                   "force" if args.force else "keep"]
    else:
        command = [
            find_rscript(), SCRIPT, process_r, ROOT, OUT_DIR,
            ",".join(map(str, args.models)), ",".join(args.outcomes), "force" if args.force else "keep",
        ]
    print("Running:", " ".join(command[:2]), "...", flush=True)
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
