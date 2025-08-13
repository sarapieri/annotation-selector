import os
import json
import sys
from typing import Dict, List, Any

# Add the annotation_selector path to import CaptionPanopticDataset
sys.path.append('tutorials/annotation_selector')
from datasets.caption_panoptic_dataset import CaptionPanopticDataset


def main():
    """Main function to process all annotated folders"""
    annotated_folders_path = "../annotated_folders/12_08"

    if not os.path.exists(annotated_folders_path):
        print(f"Error: {annotated_folders_path} does not exist")
        return

    # Load dataset configs (case-insensitive keys)
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
            all_datasets_config = {k.lower(): v for k, v in config.get("datasets", {}).items()}
    except Exception as e:
        print(f"Error: Failed to load config.json: {e}")
        return

    # Get all subfolders
    subfolders = [d for d in os.listdir(annotated_folders_path)
                  if os.path.isdir(os.path.join(annotated_folders_path, d))]

    if not subfolders:
        print(f"No subfolders found in {annotated_folders_path}")
        return

    print(f"Found {len(subfolders)} subfolders: {subfolders}")

    # Prepare summary collection
    summary_lines: List[str] = []
    per_dataset_completeness = {}

    # Process each subfolder
    datasets = {}
    for folder_name in subfolders:
        folder_path = os.path.join(annotated_folders_path, folder_name)

        # Resolve dataset config by folder name (case-insensitive)
        dataset_key_lc = folder_name.lower()
        if dataset_key_lc not in all_datasets_config:
            print(f"Warning: Dataset '{folder_name}' not found in config.json. Skipping.")
            continue
        dataset_config = all_datasets_config[dataset_key_lc]

        # Use the original-cased config key as dataset name if available
        # Fall back to folder_name otherwise
        resolved_name = next((k for k in config.get("datasets", {}).keys() if k.lower() == dataset_key_lc), folder_name)

        # Create CaptionPanopticDataset with paths from config and captions from the subfolder
        dataset = CaptionPanopticDataset(
            resolved_name,
            dataset_config["image_dir"],
            dataset_config["ann_file"],
            dataset_config["mask_dir"],
            caption_dir=folder_path,
        )

        # Load core dataset first, then captions
        dataset.load()
        dataset.load_captions()
        datasets[folder_name] = dataset

        # Do not print all captions
        # dataset.print_captions()

        # Basic caption statistics (keep console minimal)
        dataset.print_caption_statistics()

        # Compute completeness without printing per-item to console
        completeness = dataset.check_caption_completeness()
        per_dataset_completeness[folder_name] = completeness

        # Append per-dataset section to summary
        summary_lines.append(f"=== Dataset: {resolved_name} (folder: {folder_name}) ===")
        summary_lines.append(f"Total images with captions: {completeness['total_images_with_captions']}")
        summary_lines.append(f"Images with complete captions: {completeness['images_with_complete_captions']}")
        summary_lines.append(f"Images with incomplete captions: {completeness['images_with_incomplete_captions']}")
        if completeness['images_with_incomplete_captions'] > 0:
            summary_lines.append("Image IDs with incomplete captions (missing some mask IDs):")
            for frame_key in completeness['incomplete_image_ids']:
                stats = completeness['completeness_stats'][frame_key]
                summary_lines.append(f"  - {frame_key} | total_masks={stats['total_masks']} | missing={stats['missing_mask_ids']}")
                # Add caption preview for incomplete cases
                caption_preview = stats['caption_text'][:200] + "..." if len(stats['caption_text']) > 200 else stats['caption_text']
                summary_lines.append(f"    Caption preview: {caption_preview}")
                summary_lines.append("")
        
        # Log unmatched files (files that couldn't be matched to dataset frames)
        unmatched_count = dataset.caption_load_summary.get('unmatched_files', 0)
        if unmatched_count > 0:
            summary_lines.append(f"Unmatched caption files (no corresponding dataset frame found): {unmatched_count}")
            # Get the unmatched filenames that were captured during loading
            unmatched_filenames = dataset.caption_load_summary.get('unmatched_filenames', [])
            if unmatched_filenames:
                summary_lines.append("Unmatched caption filenames:")
                for filename in unmatched_filenames[:20]:  # Limit to first 20 for readability
                    summary_lines.append(f"  - {filename}")
                if len(unmatched_filenames) > 20:
                    summary_lines.append(f"  ... and {len(unmatched_filenames) - 20} more")
                summary_lines.append("")
            summary_lines.append("")
        
        # Log other loading issues
        if dataset.caption_load_summary.get('read_errors', 0) > 0:
            summary_lines.append(f"Caption files with read/parse errors: {dataset.caption_load_summary['read_errors']}")
        if dataset.caption_load_summary.get('missing_text_field', 0) > 0:
            summary_lines.append(f"Caption files missing 'text' field: {dataset.caption_load_summary['missing_text_field']}")
        if dataset.caption_load_summary.get('invalid_caption_type', 0) > 0:
            summary_lines.append(f"Caption files with invalid caption type: {dataset.caption_load_summary['invalid_caption_type']}")
        if dataset.caption_load_summary.get('non_numeric_ids', 0) > 0:
            summary_lines.append(f"Caption files with non-numeric IDs: {dataset.caption_load_summary['non_numeric_ids']}")
        if dataset.caption_load_summary.get('duplicate_mappings', 0) > 0:
            summary_lines.append(f"Duplicate caption mappings skipped: {dataset.caption_load_summary['duplicate_mappings']}")
        if dataset.caption_load_summary.get('empty_caption_files', 0) > 0:
            summary_lines.append(f"Caption files with empty captions: {dataset.caption_load_summary['empty_caption_files']}")
        
        summary_lines.append("")

        print("\n" + "="*80 + "\n")

    # Print overall summary to console (concise)
    print("=== OVERALL SUMMARY ===")
    total_captions = sum(len(dataset.captions) for dataset in datasets.values())
    total_complete_captions = 0
    total_incomplete_captions = 0

    for dataset_name, dataset in datasets.items():
        completeness_data = per_dataset_completeness[dataset_name]
        total_complete_captions += completeness_data['images_with_complete_captions']
        total_incomplete_captions += completeness_data['images_with_incomplete_captions']

    print(f"Total datasets: {len(datasets)}")
    print(f"Total captions loaded across all datasets: {total_captions}")
    print(f"Total images with complete captions: {total_complete_captions}")
    print(f"Total images with incomplete captions: {total_incomplete_captions}")
    if total_captions > 0:
        overall_completeness = (total_complete_captions / total_captions) * 100
        print(f"Overall caption completeness rate: {overall_completeness:.1f}%")

    for dataset_name, dataset in datasets.items():
        completeness_data = per_dataset_completeness[dataset_name]
        print(f"{dataset_name}: {len(dataset.captions)} captions loaded, {completeness_data['images_with_complete_captions']} complete, {completeness_data['images_with_incomplete_captions']} incomplete")

    # Write summary to file
    summary_path = os.path.join(os.path.dirname(__file__), "verification_summary.txt")

    overall_lines = [
        "=== OVERALL SUMMARY ===",
        f"Total datasets: {len(datasets)}",
        f"Total captions loaded across all datasets: {total_captions}",
        f"Total images with complete captions: {total_complete_captions}",
        f"Total images with incomplete captions: {total_incomplete_captions}",
    ]
    if total_captions > 0:
        overall_lines.append(f"Overall caption completeness rate: {overall_completeness:.1f}%")
    overall_lines.append("")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(overall_lines + summary_lines))

    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
