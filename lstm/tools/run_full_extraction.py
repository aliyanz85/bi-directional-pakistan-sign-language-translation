import os
import sys

# Ensure project root is on sys.path so local package imports work
sys.path.insert(0, os.getcwd())

from models.training.extract_landmarks import process_dataset


if __name__ == '__main__':
    # Call without selected_classes to process all folders in videosDataset
    process_dataset()
