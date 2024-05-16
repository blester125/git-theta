"""Install the git-theta-merge-cli plugin."""

import ast
import os

from setuptools import setup


def get_version(file_name: str, version_variable: str = "__version__") -> str:
    """Find the version by walking the AST to avoid duplication.

    Parameters
    ----------
    file_name : str
        The file we are parsing to get the version string from.
    version_variable : str
        The variable name that holds the version string.

    Raises
    ------
    ValueError
        If there was no assignment to version_variable in file_name.

    Returns
    -------
    version_string : str
        The version string parsed from file_name_name.
    """
    with open(file_name) as f:
        tree = ast.parse(f.read())
        # Look at all assignment nodes that happen in the ast. If the variable
        # name matches the given parameter, grab the value (which will be
        # the version string we are looking for).
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if node.targets[0].id == version_variable:
                    return node.value.s
    raise ValueError(
        f"Could not find an assignment to {version_variable} " f"within '{file_name}'"
    )


with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()


setup(
    name="git_theta_merge_cli",
    version=get_version("git_theta_merge_cli/__init__.py"),
    description="Variadic Merge Plugins for the git theta from the CLI.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Brian Lester",
    author_email="blester125@gmail.com",
    url="https://github.com/r-three/git-theta/tree/main/plugins/merge_cli",
    install_requires=[
        "git_theta",
        "torch",
    ],
    extras_require={
        "test": ["pytest"],
    },
    packages=["git_theta_merge_cli"],
    entry_points={
        "git_theta.plugins.merges": [
            "variadic-average = git_theta_merge_cli.merges.average:VariadicAverage",
            "average = git_theta_merge_cli.merges.average:VariadicAverage",
            "variadic-average-gpu = git_theta_merge_cli.merges.average:VariadicAverageGPU",
            "average-gpu = git_theta_merge_cli.merges.average:VariadicAverageGPU",
            "fisher = git_theta_merge_cli.merges.fisher:FisherMerge",
            "fisher-gpu = git_theta_merge_cli.merges.fisher:FisherMergeGPU",
            "mats = git_theta_merge_cli.merges.mats:MaTS",
            "mats-gpu = git_theta_merge_cli.merges.mats:MaTSGPU",
            "covariance-mats = git_theta_merge_cli.merges.mats:CovarianceMaTS",
            "covariance-mats-gpu = git_theta_merge_cli.merges.mats:CovarianceMaTSGPU",
            "reg-mean = git_theta_merge_cli.merges.reg_mean:RegMean",
            "reg-mean-gpu = git_theta_merge_cli.merges.reg_mean:RegMeanGPU",
            "ties = git_theta_merge_cli.merges.ties:TIES",
            "ties-gpu = git_theta_merge_cli.merges.ties:TIESGPU",
            "task-arithmetic = git_theta_merge_cli.merges.task_arithmetic:TaskArithmetic",
            "task-arithmetic-gpu = git_theta_merge_cli.merges.task_arithmetic:TaskArithmeticGPU",
            "dare-task-arithmetic = git_theta_merge_cli.merges.task_arithmetic:DARETaskArithmetic",
            "dare-task-arithmetic-gpu = git_theta_merge_cli.merges.task_arithmetic:DARETaskArithmeticGPU",
            "lora-variadic-average = git_theta_merge_cli.merges.average:VariadicAverageLoRA",
            "lora-average = git_theta_merge_cli.merges.average:VariadicAverageLoRA",
            "lora-variadic-average-gpu = git_theta_merge_cli.merges.average:VariadicAverageLoRAGPU",
            "lora-average-gpu = git_theta_merge_cli.merges.average:VariadicAverageLoRAGPU",
            "lora-fisher = git_theta_merge_cli.merges.fisher:FisherMergeLoRA",
            "lora-fisher-gpu = git_theta_merge_cli.merges.fisher:FisherMergeLoRAGPU",
            "lora-mats = git_theta_merge_cli.merges.mats:MaTSLoRA",
            "lora-mats-gpu = git_theta_merge_cli.merges.mats:MaTSLoRAGPU",
            "lora-covariance-mats = git_theta_merge_cli.merges.mats:CovarianceMaTSLoRA",
            "lora-covariance-mats-gpu = git_theta_merge_cli.merges.mats:CovarianceMaTSLoRAGPU",
            "lora-reg-mean = git_theta_merge_cli.merges.reg_mean:RegMeanLoRA",
            "lora-reg-mean-gpu = git_theta_merge_cli.merges.reg_mean:RegMeanLoRAGPU",
            "lora-ties = git_theta_merge_cli.merges.ties:TIESLoRA",
            "lora-ties-gpu = git_theta_merge_cli.merges.ties:TIESLoRAGPU",
            "lora-task-arithmetic = git_theta_merge_cli.merges.task_arithmetic:TaskArithmeticLoRA",
            "lora-task-arithmetic-gpu = git_theta_merge_cli.merges.task_arithmetic:TaskArithmeticLoRAGPU",
            "lora-dare-task-arithmetic = git_theta_merge_cli.merges.task_arithmetic:DARETaskArithmeticLoRA",
            "lora-dare-task-arithmetic-gpu = git_theta_merge_cli.merges.task_arithmetic:DARETaskArithmeticLoRAGPU",
        ],
        "console_scripts": [
            "git-theta-merge-cli = git_theta_merge_cli.scripts.git_theta_merge_cli:main",
            "git-theta-merge-majority-sign = git_theta_merge_cli.scripts.majority_sign:main",
        ],
    },
)
