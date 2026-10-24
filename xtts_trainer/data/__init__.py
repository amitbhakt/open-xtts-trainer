"""Dataset contract: the dataset.jsonl schema and the validator every stage runs before training."""

from xtts_trainer.data.schema import (
    DATASET_FILENAME,
    DatasetFormatError,
    DatasetRecord,
    load_dataset,
    write_dataset,
)
from xtts_trainer.data.validate import (
    DatasetValidationError,
    ValidationConfig,
    ValidationReport,
    load_validated_dataset,
    validate_dataset,
)

__all__ = [
    "DATASET_FILENAME", "DatasetFormatError", "DatasetRecord", "load_dataset", "write_dataset",
    "DatasetValidationError", "ValidationConfig", "ValidationReport", "load_validated_dataset", "validate_dataset",
]
