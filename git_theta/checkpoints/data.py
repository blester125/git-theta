"""'Checkpoint' for datasets."""

import json

from file_or_name import file_or_name

from git_theta import utils
from git_theta.checkpoints import Checkpoint


class JsonlDataCheckpoint(Checkpoint):
    """Class for r/w data in jsonl."""

    name: str = "data-jsonl"

    @classmethod
    @file_or_name(checkpoint_path="r")
    def load(cls, checkpoint_path: str):
        data = [json.loads(l) for l in checkpoint_path if l]
        return {d["id"]: d for d in data}

    @classmethod
    def from_framework(cls, model_dict):
        return cls(model_dict)

    def to_framework(self):
        return self

    @file_or_name(checkpoint_path="w")
    def save(self, checkpoint_path):
        checkpoint_dict = self.to_framework()
        checkpoint_path.write(
            "\n".join(json.dumps(v) for v in self.values()).encode("utf-8")
        )

    def flatten(self):
        return utils.flatten(self, is_leaf=lambda v: isinstance(v, dict))

    @classmethod
    def leaf_equal(cls, x, y):
        return x == y
