"""Inspect supported restoration references without importing the training runtime."""

import argparse
import json

from .reference import load_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("adamw", "muon"), required=True)
    parser.add_argument("--rank", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    ref = load_reference(args.case, args.rank)
    print(
        json.dumps(
            {
                k: ref[k]
                for k in (
                    "case",
                    "rank",
                    "run_id",
                    "component_sha256",
                    "cold_sha256",
                    "warm_sha256",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
