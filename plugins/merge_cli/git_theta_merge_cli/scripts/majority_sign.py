"""Calculate the majority sign from trimmed TIES vectors."""

import argparse
import asyncio
import logging

import torch
from git_theta_merge_cli import utils
from git_theta_merge_cli.merges import reductions

import git_theta
from git_theta import async_utils, git_utils, metadata, params, updates

git_theta.scripts.configure_logging("majority-sign")


parser = argparse.ArgumentParser(
    description="Calculate the majority sign of trimmed TIES vectors."
)
parser.add_argument(
    "--ties",
    nargs="+",
    required=True,
    default=[],
    help=("A list of TIES trimmed task vectors for merging."),
)
parser.add_argument(
    "--limit-concurrency",
    default=10,
    type=int,
    help="The maximum number of parameter to process concurrently.",
)


def main():
    args = parser.parse_args()

    repo = git_utils.get_git_repo()
    models = [utils.load_metadata(repo, path).flatten() for path in args.ties]

    logger = logging.getLogger("git_theta")
    sign_calc = reductions.Sign()
    global_majority_sign = sign_calc.run_merge(models, [], {})

    print(f"Global Majority Sign is: {global_majority_sign}")


if __name__ == "__main__":
    main()
