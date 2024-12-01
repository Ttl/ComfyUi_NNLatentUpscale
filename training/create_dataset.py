import os
import random
import tarfile
from pathlib import Path
from typing import List, Tuple


def create_train_test_split(
    source_directory: str,
    output_directory: str,
    test_ratio: float = 0.1,
    shard_size: int = 250,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    random.seed(seed)

    train_dir = os.path.join(output_directory, "train")
    test_dir = os.path.join(output_directory, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    image_files = []
    valid_extensions = (".jpg", ".jpeg", ".png", ".tiff", ".bmp")
    for path in Path(source_directory).rglob("*"):
        if path.suffix.lower() in valid_extensions:
            image_files.append(path)

    random.shuffle(image_files)
    split_idx = int(len(image_files) * (1 - test_ratio))
    train_files = image_files[:split_idx]
    test_files = image_files[split_idx:]

    def create_shards(files: List[Path], output_dir: str, prefix: str) -> List[str]:
        shard_paths = []

        for shard_idx in range(0, len(files), shard_size):
            shard_files = files[shard_idx : shard_idx + shard_size]
            shard_name = f"{prefix}_{shard_idx//shard_size:06d}.tar"
            shard_path = os.path.join(output_dir, shard_name)
            shard_paths.append(shard_path)

            with tarfile.open(shard_path, "w") as tar:
                for idx, img_path in enumerate(shard_files):
                    try:
                        # Use different prefixes for train and test files
                        arcname = f"{prefix}_{idx:08d}{img_path.suffix}"
                        tar.add(img_path, arcname=arcname)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        continue

        return shard_paths

    train_shards = create_shards(train_files, train_dir, "train")
    test_shards = create_shards(test_files, test_dir, "test")

    print(f"Dataset split complete:")
    print(f"Total images: {len(image_files)}")
    print(
        f"Training images: {len(train_files)} ({len(train_files)/len(image_files)*100:.1f}%)"
    )
    print(
        f"Testing images: {len(test_files)} ({len(test_files)/len(image_files)*100:.1f}%)"
    )
    print(f"Training shards: {len(train_shards)}")
    print(f"Testing shards: {len(test_shards)}")

    return train_shards, test_shards


def verify_split_integrity(train_shards: List[str], test_shards: List[str]) -> None:
    train_files = set()
    test_files = set()

    def check_shard(shard_path: str) -> set:
        files = set()
        with tarfile.open(shard_path, "r") as tar:
            members = tar.getmembers()
            if len(members) == 0:
                raise ValueError(f"Empty shard found: {shard_path}")

            for member in members:
                f = tar.extractfile(member)
                if f is None:
                    raise ValueError(
                        f"Unreadable file in shard {shard_path}: {member.name}"
                    )
                files.add(member.name)
        return files

    print("Verifying train shards...")
    for shard in train_shards:
        train_files.update(check_shard(shard))

    print("Verifying test shards...")
    for shard in test_shards:
        test_files.update(check_shard(shard))

    overlap = train_files.intersection(test_files)
    if overlap:
        raise ValueError(
            f"Found {len(overlap)} duplicate files between train and test sets!"
        )

    print("Verification complete:")
    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")
    print("No duplicates found between train and test sets")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir", type=str, default="path/to/your/high_quality_images"
    )
    parser.add_argument("--output_dir", type=str, default="path/to/output/webdataset")
    parser.add_argument("--shard_size", type=int, default=250)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Create the split
    train_shards, test_shards = create_train_test_split(
        source_directory=args.source_dir,
        output_directory=args.output_dir,
        test_ratio=args.test_ratio,
        shard_size=args.shard_size,
        seed=args.seed,
    )

    # Verify the split
    verify_split_integrity(train_shards, test_shards)
