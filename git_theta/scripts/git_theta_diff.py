"""Tool for creating diffs with git-theta."""

import argparse
import json
import sys
import textwrap
from typing import Optional

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

import numpy as np
from colorama import Fore, Style

import git_theta
from git_theta import checkpoints, metadata, utils

git_theta.scripts.configure_logging("git-theta-diff")


def parse_args():
    parser = argparse.ArgumentParser(description="git-theta diff program")
    parser.add_argument("path", help="path to file being diff-ed")

    parser.add_argument(
        "old_checkpoint", help="file that old version of checkpoint can be read from"
    )
    parser.add_argument("old_hex", help="SHA-1 hash of old version of checkpoint")
    parser.add_argument("old_mode", help="file mode for old version of checkpoint")

    parser.add_argument(
        "new_checkpoint", help="file that new version of checkpoint can be read from"
    )
    parser.add_argument("new_hex", help="SHA-1 hash of new version of checkpoint")
    parser.add_argument("new_mode", help="file mode for new version of checkpoint")

    args = parser.parse_args()
    return args


class DiffSummarizer:
    """Base Class the summarizes differences between things."""

    def diff(self, new, old) -> str:
        raise NotImplementedError


class TensorDiff(DiffSummarizer):
    """Difference between two Tensors."""

    def diff(self, new, old):
        # TODO: Add more useful diff information between tensor values like
        # size, dtype, change in norm, etc.
        return ""


class JsonDiff(DiffSummarizer):
    def diff(self, new, old):
        new = color_string(f"+{json.dumps(new)}", Fore.GREEN) if new is not None else ""
        old = color_string(f"-{json.dumps(old)}", Fore.RED) if old is not None else ""
        join = "\n" if new and old else ""
        return f"{new}{join}{old}"


def get_diff_handler(diff_type: Optional[str] = None) -> DiffSummarizer:
    diff_type = diff_type or utils.EnvVarConstants.DIFF_TYPE
    discovered_plugins = entry_points(group="git_theta.plugins.diffs")
    return discovered_plugins[diff_type].load()()


def color_string(s, color):
    return "\n".join([f"{color}{s_}" if color else s_ for s_ in s.split("\n")])


def bold_string(s):
    return f"{Style.BRIGHT}{s}"


def print_formatted(s, indent=0, color=None, bold=False):
    if indent:
        s = "\n".join(
            textwrap.wrap(
                s,
                initial_indent=" " * 4 * indent,
                subsequent_indent=" " * 4 * (indent + 1),
            )
        )
    if color:
        s = color_string(s, color)
    if bold:
        s = bold_string(s)
    print(s)


def print_header(header, indent=0, color=None):
    print_formatted(header, indent=indent, color=color, bold=True)
    print_formatted("-" * len(header), indent=indent, color=color, bold=True)


def print_added_params_summary(added, indent=0, color=None):
    if added:
        print_header("ADDED PARAMETER GROUPS", indent=indent, color=color)
        for flattened_group, param in added.flatten().items():
            group = "/".join(flattened_group)
            print_formatted(group, indent=indent, color=color)
            print_formatted(get_diff_handler().diff(param, None), indent=indent)
        print_formatted("\n")


def print_removed_params_summary(removed, indent=0, color=None):
    if removed:
        print_header("REMOVED PARAMETER GROUPS", indent=indent, color=color)
        for flattened_group, param in removed.flatten().items():
            group = "/".join(flattened_group)
            print_formatted(group, indent=indent, color=color)
            print_formatted(get_diff_handler().diff(None, param), indent=indent)
        print_formatted("\n")


def print_modified_params_summary(modified, indent=0, color=None):
    if modified:
        print_header("MODIFIED PARAMETER GROUPS", indent=indent, color=color)
        for flattened_group, params in utils.flatten(
            modified, is_leaf=lambda v: isinstance(v, tuple)
        ).items():
            group = "/".join(flattened_group)
            print_formatted(group, indent=indent, color=color)
            print_formatted(get_diff_handler().diff(*params), indent=indent)
        print_formatted("\n")


def main():
    args = parse_args()
    checkpoint_handler = checkpoints.get_checkpoint_handler()
    old_checkpoint = checkpoint_handler.from_file(args.old_checkpoint)
    new_checkpoint = checkpoint_handler.from_file(args.new_checkpoint)
    added, removed, modified = checkpoint_handler.diff(new_checkpoint, old_checkpoint)

    print_added_params_summary(added, indent=0, color=Fore.GREEN)
    print_removed_params_summary(removed, indent=0, color=Fore.RED)
    print_modified_params_summary(modified, indent=0, color=Fore.YELLOW)
