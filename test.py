import os
from collections import defaultdict


def map_filenames(directory: str) -> dict:
    file_map = defaultdict(list)
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            key = filename.split("_")[0]
            file_map[key].append(filename)
    return dict(file_map)


if __name__ == "__main__":
    directory = "gdsfactory/components"
    file_map = map_filenames(directory)
    for key, files in file_map.items():
        if len(files) > 1:
            print(f"{key}", end=", ")
