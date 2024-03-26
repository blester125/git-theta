"""An interface to merging where you can run it on multiple models with different paths."""

import argparse
import asyncio
import collections
import functools
import io
import json
import logging
import sys

import git_theta
from git_theta import async_utils, git_utils, merges, metadata

git_theta.scripts.configure_logging("git-theta-merge")


parser = argparse.ArgumentParser(
    description=(
        "git-theta-merge, but eaiser. Arguments to the merge method "
        "can be passed with --x:${name}=${value}"
    ),
)
parser.add_argument(
    "--models",
    nargs="+",
    required=True,
    help=(
        "A list of relative paths for models to merge. Parameter names "
        + "should match in all cases."
    ),
)
parser.add_argument(
    "--aux_data",
    nargs="*",
    required=False,
    default=[],
    help=(
        "A list of auxiliary data to use when merging, things like the "
        + "FisherMatrix. Should have one value for each model that is being merged."
    ),
)
parser.add_argument(
    "--output", required=True, help="The name of the resulting model checkpoint file."
)
# TODO: Is there a clean way to enumerate the plugins for the help menu?
parser.add_argument(
    "--merge",
    default="variadic-average",
    help="The merge method to apply to the checkpoints.",
)
parser.add_argument(
    "--ancestor",
    help=(
        "The path to the original model when models to be merged are "
        + "finetuned from the same checkpoint."
    ),
)
parser.add_argument(
    "--limit-concurrency",
    default=10,
    type=int,
    help="The maximum number of parameters to merge concurrently.",
)
parser.add_argument("--message", "-m", default=None, help="Custom commit message.")
parser.add_argument(
    "--test_run",
    type=int,
    default=None,
    help=(
        "Only merge a small number of parameters for testing. The lm head "
        + "will also be merged to test large parameter handling."
    ),
)


def load_metadata(repo, path):
    """Load model metadata from git."""
    # We could also support merging files from different tags/commits by parsing
    # that from the paths and not forcing HEAD?
    metadata_blob = git_utils.get_file_version(repo, path, "HEAD")
    return metadata.Metadata.from_file(metadata_blob.data_stream)


async def merge(param_name, to_merge, merge_method):
    """Run the merge.

    We use this function as it has the signature used in run_map.
    """
    params, aux_data, ancestor = to_merge
    # This is all metadata at the moment.
    logger = logging.getLogger("git_theta")
    logger.info(f"Merging {'/'.join(param_name)}.")
    # If the parameter metadata is the same for all models, just return the original metadata.
    if all(params[0] == p for p in params[1:]):
        logger.debug(
            f"Skipping Merge of {'/'.join(param_name)} as it is the same across models."
        )
        return (param_name, params[0])
    return (param_name, await merge_method(param_name, params, aux_data, ancestor))


def _infer_numeric_or_str(value: str):
    """Parse a cli argument into an int, float or leave as a str."""
    for func in (int, float):
        try:
            # int(float) wouldn't throw a ValueError but int(str(float)) will.
            return func(value)
        except ValueError:
            continue
    return value


def parse_extra_args(args, prefix: str = "x"):
    """Allow users to pass arbitrary arguments in the format --x:${name}=${value}.

    Supports passing a list via repeated --x:${name}=${value2}.
    """
    logger = logging.getLogger("git_theta")
    prefix = f"--{prefix}:"
    parser = argparse.ArgumentParser()
    # Find all args that have the prefix.
    known_args = set(filter(lambda x: prefix in x, args))
    # Build a parse with these new prefixed arguments.
    for key in known_args:
        parser.add_argument(
            key.split("=")[0], action="append", type=_infer_numeric_or_str
        )
    for key in args:
        if key not in known_args and key.startswith("--"):
            # TODO: Should we do more than use a warning? This allows typos in
            #       defined arguments to be allowed and ignored.
            logger.warning(f"Unexpected argument: {key}, Skipping")
    args = parser.parse_known_args(args)[0]
    # Remove the --x: part of the flags, delistify single list values, and convert to dict.
    return {k.split(":")[1]: v[0] if len(v) == 1 else v for k, v in vars(args).items()}


def main():
    # Parse user supplied merge method arguments.
    args, extra_args = parser.parse_known_args()
    extra_args = parse_extra_args(extra_args)

    repo = git_utils.get_git_repo()
    # Load the metadata for all models from git
    models = [load_metadata(repo, path).flatten() for path in args.models]

    # Load any auxiliary data, should have matching keys as the main model and be stored in git too?
    aux_data = [load_metadata(repo, path).flatten() for path in args.aux_data]
    if aux_data:
        if len(aux_data) != len(models):
            raise ValueError(
                "If auxiliary data is provided, it needs to be provided "
                f"for all models. Got {len(models)} models and {len(aux_data)} aux data."
            )

    # Load the ancestor pre-trained checkpoint if it is needed.
    ancestor = load_metadata(repo, args.ancestor).flatten() if args.ancestor else {}

    # Load the merge handler we want to use.
    logger = logging.getLogger("git_theta")
    logger.info(f"Creating Merge Handler {args.merge} with args: {extra_args}")
    merge_method = merges.get_merge_handler(args.merge)(**extra_args)

    # A chance to change the whole model at once, for things like combining LoRA params.
    logger.info(f"Adjusting parameter names.")
    models = [merge_method.rewrite_checkpoint(m) for m in models]
    aux_data = [merge_method.rewrite_checkpoint(ad) for ad in aux_data]
    ancestor = merge_method.rewrite_checkpoint(ancestor) if ancestor else ancestor

    # Assumes we have matching parameter names.
    # This is a mapping from parameter name to a tuple of lists representing
    #   (parameters_md, aux data_md, ancestor_md) for each model.
    # It groups together everything we need for the merge.
    # If ancestor wasn't provided, it is a empty dict so `.get` gives a None.
    # If aux_data isn't provided at all, it is an empty list so we don't call .get
    #   If aux_data isn't provided for a parameter in the model, `.get` gives a None.
    merged_model = {
        p: ([m[p] for m in models], [aux.get(p) for aux in aux_data], ancestor.get(p))
        for p in models[0]
    }
    if args.test_run:
        # TODO: Remove this? Or make the extra param to merge configurable?
        # Only merge a few parameters, also include the lm_head as it is very large and
        # often requires special code. We want to make sure merging it works.
        merged_model = {
            k: v
            for i, (k, v) in enumerate(merged_model.items())
            if i < args.test_run or k == ("transformer.lm_head.weight",)
        }
        logger.info(f"Only merging {len(merged_model)} parameters")

    logger.info("Merging the model")
    # Merge the model while limiting how many parameters we have in memory at once.
    merged_model = async_utils.run(
        async_utils.run_map(
            merged_model,
            functools.partial(merge, merge_method=merge_method),
            max_concurrency=args.limit_concurrency,
        )
    )

    logger.info("Comitting Merged Model")
    # Setup the the repo to track the model.
    repo = git_utils.get_git_repo()
    # Convert the output path to be relative to the repo, otherwise you can commit
    # an absolute path to git and corrupt the metadata.
    model_path = git_utils.get_relative_path_from_root(repo, args.output)
    gitattributes_file = git_utils.get_gitattributes_file(repo)
    gitattributes = git_utils.read_gitattributes(gitattributes_file)
    new_gitattributes = git_utils.add_theta_to_gitattributes(gitattributes, model_path)
    git_utils.write_gitattributes(gitattributes_file, new_gitattributes)
    git_utils.add_file(gitattributes_file, repo)

    # Convert the merged model from a dict of metadata to a metadata object.
    merged_model = metadata.Metadata(**merged_model).unflatten()
    # Convert the metadata object to a string.
    with io.StringIO() as f:
        merged_model.write(f)
        merged_model = f.getvalue()
    # Add the metadata to git, the parameters have already been stored in git-lfs
    blob = git_theta.git_utils.make_blob(repo, merged_model, model_path)
    # Add the blob to the index.
    repo.index.add([blob])
    # Commit the model to git to save it going forward.
    # We could smudge here, but that would materalize the merged model in memory.
    msg = (
        f"Committing the merged model to {model_path}"
        if args.message is None
        else args.message
    )
    if sys.platform in ("win32", "cygwin"):
        repo.git.commit(m=msg)
        sha = repo.commit("HEAD")
    else:
        sha = repo.index.commit(msg)
    logger.info("Checking out Merged Model")
    # Checkout the new commit to ensure the merged model is on disk.
    repo.git.checkout("HEAD")
    # Only hard checkout the model file, doing a reset would blow away any other changes.
    with repo.git.custom_environment(
        # Set the max concurrency during the smudge to be the same as merging.
        # You could probably get away with more as you aren't loading multiple
        # versions of the model, but this ensures there is enough memory.
        **{"GIT_THETA_MAX_CONCURRENCY": str(args.limit_concurrency)}
    ):
        repo.git.checkout("HEAD", "--", model_path)


if __name__ == "__main__":
    main()
