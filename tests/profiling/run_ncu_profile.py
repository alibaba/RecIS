import argparse

from recis.utils.profiler.ncu_profiler import run_ncu_profiling


def main():
    parser = argparse.ArgumentParser(description="Run ncu profiling for recis ops")
    parser.add_argument("--script", required=True, help="Training script to profile")
    parser.add_argument("--args", default="", help="Arguments for the training script")
    args = parser.parse_args()

    run_ncu_profiling(script=args.script, args=args.args)


if __name__ == "__main__":
    main()
