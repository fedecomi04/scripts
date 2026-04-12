from nerfstudio.engine.trainer import Trainer


class NoSaveTrainer(Trainer):
    """Trainer that skips all checkpoint saving — useful during rapid iteration/testing."""

    def save_checkpoint(self, step: int) -> None:
        pass
