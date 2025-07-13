import os
import json
import argparse
import shutil
import sys
from tqdm import tqdm
from datasets.panoptic_dataset import PanopticDataset


def process_selection_file(selection_file, all_datasets_config):
    print(f"\n Processing: {selection_file}")

    # 1. Parse dataset name
    base_name = os.path.basename(selection_file)
    if base_name.startswith("selected_") and base_name.endswith(".json"):
        dataset_name = base_name[len("selected_"):-len(".json")].lower()
    else:
        raise ValueError(f"Invalid file format '{base_name}'. Expected 'selected_<dataset>.json'.")

    # 2. Load dataset config
    if dataset_name not in all_datasets_config:
        raise KeyError(f"Dataset '{dataset_name}' not found in config.json.")

    dataset_config = all_datasets_config[dataset_name]

    # 3. Initialize dataset
    dataset = PanopticDataset(name=dataset_name, **dataset_config)
    dataset.load()

    # 4. Load selected file list
    try:
        with open(selection_file, "r") as f:
            selection_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load or parse selection file '{selection_file}': {e}")

    if isinstance(selection_data, dict):
        selected_files = [f for files in selection_data.values() for f in files]
    elif isinstance(selection_data, list):
        selected_files = selection_data
    else:
        raise ValueError(f"Unsupported format in selection file '{selection_file}'")

    if not selected_files:
        raise ValueError(f"No files found in selection file: {selection_file}")

    print(f" Found {len(selected_files)} files to export for '{dataset_name}'")

    # 5. Output directory
    output_base_dir = "exports"
    dataset_export_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(dataset_export_dir, exist_ok=True)

    for fname in tqdm(selected_files, desc=f"Exporting {dataset_name}"):
        image_id = os.path.splitext(fname)[0]
        if dataset.is_video_dataset:
            video_id = dataset.video_map.get(fname)
            if not video_id:
                raise KeyError(f"Missing video_id mapping for '{fname}'")
            image_id = f"{video_id}_{image_id}"

        output_item_dir = os.path.join(dataset_export_dir, image_id)
        os.makedirs(output_item_dir, exist_ok=True)

        # Load visualized data
        original_qimg, mask_qimg, labels, _ = dataset.load_image(fname)

        if original_qimg is None or mask_qimg is None:
            raise RuntimeError(f"Failed to load visual data for '{fname}'.")

        # a) Save original image (try loading from disk first)
        jpg_fname = fname.replace(".png", ".jpg")
        if dataset.is_video_dataset:
            video_id = dataset.video_map.get(fname)
            original_image_path = os.path.join(dataset.image_dir, video_id, jpg_fname)
        else:
            original_image_path = os.path.join(dataset.image_dir, jpg_fname)

        if not os.path.isfile(original_image_path):
            raise FileNotFoundError(f"Original image not found: {original_image_path}")

        shutil.copy(original_image_path, os.path.join(output_item_dir, "original.jpg"))

        # b) Save overlay image
        mask_output_path = os.path.join(output_item_dir, "overlay.png")
        if not mask_qimg.save(mask_output_path):
            raise IOError(f"Failed to save mask overlay to {mask_output_path}")

        # c) Save label list
        labels_txt_path = os.path.join(output_item_dir, "labels.txt")
        with open(labels_txt_path, "w") as f:
            f.write("\n".join(labels[:-1]))  # Exclude coverage

    print(f" Finished exporting {len(selected_files)} items to '{dataset_export_dir}'")


def main():
    parser = argparse.ArgumentParser(
        description="Export selected annotations into folders with overlays and label files."
    )
    parser.add_argument(
        "selection_files",
        nargs="*",
        help="Optional path(s) to selection JSON files. If empty, scans the selected_annotations/ folder."
    )
    args = parser.parse_args()

    # Load config.json
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
            all_datasets_config = {k.lower(): v for k, v in config.get("datasets", {}).items()}
    except Exception as e:
        sys.exit(f" Failed to load config.json: {e}")

    # Find selection files
    selection_dir = "selected_annotations"
    if args.selection_files:
        files_to_process = args.selection_files
    else:
        if not os.path.isdir(selection_dir):
            sys.exit(f" Selection directory not found: {selection_dir}")
        files_to_process = [
            os.path.join(selection_dir, f)
            for f in os.listdir(selection_dir)
            if f.startswith("selected_") and f.endswith(".json")
        ]

    if not files_to_process:
        sys.exit(" No selection files found to process.")

    # Process each selection file
    for selection_file in files_to_process:
        try:
            process_selection_file(selection_file, all_datasets_config)
        except Exception as e:
            print(f" Error processing {selection_file}: {e}")
            sys.exit(1)

    print("\n Export process completed successfully.")


if __name__ == "__main__":
    main()
