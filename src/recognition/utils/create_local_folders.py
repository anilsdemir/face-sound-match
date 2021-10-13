"""
Script to be used to create local folders for running the project successfully.
You need to run this script only once when you fresh clone the repo.
If the script changes you might need to run it again to create new directories.
"""

import os
import sys
import logging

from pytorch_classification.config import (
    BASE_DIR,
    LOCAL_DIR,
    INPUT_DIR,
    DATA_DIR,
    MODELS_DIR,
    LOGGING_DIR,
    OUTPUT_DIR,
)

folders_to_be_created = [
    BASE_DIR,
    LOCAL_DIR,
    INPUT_DIR,
    DATA_DIR,
    MODELS_DIR,
    LOGGING_DIR,
    OUTPUT_DIR,
]

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    for folder in folders_to_be_created:
        try:
            os.mkdir(folder)
            logger.info(f"Created folder with path: {folder}")
        except FileExistsError:
            pass