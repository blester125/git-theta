"""Calculate the majority sign from trimmed TIES vectors."""

import argparse
import asyncio
import logging

import torch
from git_theta_merge_cli.merges import utils

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


def load_metadata(repo, path):
    """Load model metadata from git."""
    # We could also support merging files from different tags/commits by parsing
    # that from the paths and not forcing HEAD?
    metadata_blob = git_utils.get_file_version(repo, path, "HEAD")
    return metadata.Metadata.from_file(metadata_blob.data_stream)


async def read_param(param_md, param_name):
    update_handler = updates.get_update_handler(param_md.theta_metadata.update_type)(
        params.get_update_serializer()
    )
    return torch.tensor(await update_handler.apply(param_md, param_name))


async def calculate_sign(param_name, param_metadata):
    logger = logging.getLogger("majority-sign")
    logger.info(f"Calculating Signs for {'/'.join(param_name)}")
    params = await asyncio.gather(*(read_param(p, param_name) for p in param_metadata))

    sum_params = utils.memory_efficient_interpolate(params)
    resolved_sign = torch.sign(sum_params)
    sign_total = torch.sum(resolved_sign).item()

    return (param_name, sign_total)


def main():
    args = parser.parse_args()

    repo = git_utils.get_git_repo()
    models = [load_metadata(repo, path).flatten() for path in args.ties]

    logger = logging.getLogger("git_theta")
    sign_calc = {p: ([m[p] for m in models]) for p in models[0]}
    sign_calc = async_utils.run(
        async_utils.run_map(
            sign_calc, calculate_sign, max_concurrency=args.limit_concurrency
        )
    )

    sign_total = sum(sign_calc.values())
    global_majority_sign = 1 if sign_total >= 0 else -1

    print(f"Global Majority Sign is: {global_majority_sign}")


if __name__ == "__main__":
    main()
