from pathlib import Path
import shutil

DEVICE_ID = "70a09b5174d07fff"
SOURCE_FOLDERS = [
    "Application data",
    "Background data",
    "Dynamic data",
    "Static data",
]


def copy_device_files(base_dir: Path, device_id: str) -> None:
    dest_root = base_dir / "70a0"
    dest_root.mkdir(exist_ok=True)

    total = 0
    for folder_name in SOURCE_FOLDERS:
        src_dir = base_dir / folder_name
        if not src_dir.exists():
            print(f"Skip missing folder: {src_dir}")
            continue

        dest_dir = dest_root / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for path in src_dir.rglob("*"):
            if path.is_file() and device_id in str(path):
                rel_path = path.relative_to(src_dir)
                target_path = dest_dir / rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target_path)
                count += 1

        total += count
        print(f"{folder_name}: copied {count} files")

    print(f"Done. Total copied files: {total}")
    print(f"Output folder: {dest_root}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    copy_device_files(base_dir, DEVICE_ID)
