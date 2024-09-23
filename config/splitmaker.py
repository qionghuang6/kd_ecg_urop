import json
import random
import os


def make_split(filename: str, seed: int) -> None:
    """
    Create and save a dataset split for the PTB-XL dataset.

    Args:
        filename (str): The output file path to save the split.
        seed (int): Random seed for reproducibility.

    This function reads the original split, shuffles the data, and creates new
    train/val/test splits with a 70/15/15 ratio.
    """
    # Read the original split file
    with open("split.json", encoding="utf-8") as f:
        data = json.load(f)

    # Set random seed for reproducibility
    random.seed(seed)

    splits: dict[str, dict[str, list[str]]] = {}

    # Get all recordings for PTB-XL dataset and shuffle them
    dataset_recordings = data["PTB-XL"]["all"][:]
    random.shuffle(dataset_recordings)

    dataset = "PTB-XL"
    splits[dataset] = {}
    splits[dataset]["all"] = dataset_recordings[:]

    # Calculate split indices (70% train, 15% val, 15% test)
    total_recordings = len(splits[dataset]["all"])
    train_end = int(0.7 * total_recordings)
    val_end = int(0.85 * total_recordings)

    # Create the splits
    splits[dataset]["train"] = splits[dataset]["all"][:train_end]
    splits[dataset]["val"] = splits[dataset]["all"][train_end:val_end]
    splits[dataset]["test"] = splits[dataset]["all"][val_end:]

    # Print split sizes for verification
    print(f"Train: {len(splits[dataset]['train'])}")
    print(f"Validation: {len(splits[dataset]['val'])}")
    print(f"Test: {len(splits[dataset]['test'])}")

    # Ensure all recordings are accounted for
    assert (
        sum(len(splits[dataset][split]) for split in ["train", "val", "test"])
        == total_recordings
    )

    # Save the new split to a file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)


if __name__ == "__main__":
    # Create 'splits' directory if it doesn't exist
    os.makedirs("splits", exist_ok=True)

    # Generate 100 different splits
    for i in range(100):
        output_file = f"splits/split_{i}.json"
        make_split(output_file, i)
        print(f"Split {i} created and saved to {output_file}")
