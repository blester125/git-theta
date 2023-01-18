"""Handle T5X checkpoints."""

from git_theta.checkpoints import Checkpoint
from t5x import checkpoints as t5x_checkpoints


class T5XCheckpoint(Checkpoint):
    @property
    def name(self):
        return "t5x"

    @classmethod
    def load(cls, checkpoint_path):
        return t5x_checkpoints.load_t5x_checkpoint(checkpoint_path)

    def save(self, checkpoint_path):
        # TODO: How to output the checkpoint when things are partitioned?
        ...
