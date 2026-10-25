"""open-xtts-trainer: Complete 3-Stage Training Suite for XTTS v2."""

__version__ = "0.1.0"
__author__ = "Amit Bhakt"


def patch_xtts_prevent_checkpoint_mutation() -> None:
    """Prevent upstream coqui-tts from mutating model.pth on disk during loading.

    Upstream Xtts._load_checkpoint re-saves model.pth in place when it encounters
    a pickled XttsConfig (Open Finding P1 in REMEDIATION_PLAN.md). This mutates
    the file from its released hash (c7ea2000...) to 3077a05b... on the first load.
    We patch it to perform the config conversion in memory without writing to disk.
    """
    try:
        from TTS.tts.models.xtts import Xtts
    except ImportError:
        return

    if getattr(Xtts, "_open_xtts_no_mutation_patched", False):
        return

    def safe_load_checkpoint(self, model_path):
        import pickle
        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.utils.io import load_fsspec
        from TTS.tts.models.xtts import _legacy_safe_globals, is_pytorch_at_least_2_4

        try:
            checkpoint = load_fsspec(model_path, map_location=torch.device("cpu"))
        except pickle.UnpicklingError:
            with _legacy_safe_globals():
                checkpoint = torch.load(
                    model_path, map_location=torch.device("cpu"), weights_only=is_pytorch_at_least_2_4()
                )
            if isinstance(checkpoint.get("config"), XttsConfig):
                checkpoint["config"].speakers = []
                checkpoint["config"] = checkpoint["config"].to_dict()
                # Omit upstream's torch.save(checkpoint, model_path) so artifacts remain immutable on disk
        return checkpoint["model"]

    Xtts._load_checkpoint = safe_load_checkpoint
    Xtts._open_xtts_no_mutation_patched = True

