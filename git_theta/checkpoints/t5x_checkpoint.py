"""Handle T5X checkpoints."""

from git_theta.checkpoints import Checkpoint


class T5XCheckpoint(Checkpoint):

    @property
    def name(self):
        return "t5x
