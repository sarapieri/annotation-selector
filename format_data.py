import os
import json
import random
from PIL import Image
from datasets.caption_panoptic_dataset import CaptionPanopticDataset
from tqdm import tqdm
import shutil
import re 

def parse_caption(caption: str):
    # Strict pattern: captures mask_ids and label text
    pattern = re.compile(r'<\s*([\d,]+)\s*:\s*([^>]+?)\s*>')
    results = []

    def replacer(match):
        ids = [int(x) for x in match.group(1).split(",")]
        text = match.group(2).strip()
        results.append({"mask_ids": ids, "txt_desc": text})

        start, end = match.span()
        before = caption[start - 1] if start > 0 else ''
        after = caption[end] if end < len(caption) else ''

        # Add a leading space only if needed
        if before not in (' ', '\t', '\n', ''):
            text = ' ' + text
        # Add a trailing space only if needed
        if after not in (' ', '\t', '\n', ''):
            text = text + ' '

        return text

    cleaned = pattern.sub(replacer, caption)
    return results, cleaned

def split_validation_data(frame_keys, test_ratio=0.7):
    """
    Split validation frame keys into test and validation sets.
    Returns (test_keys, val_keys)
    """
    random.shuffle(frame_keys)
    split_idx = int(len(frame_keys) * test_ratio)
    return frame_keys[:split_idx], frame_keys[split_idx:]

def process_dataset_split(dataset, frame_keys, split_name, ann_index_start, cap_id_start=0):
    """
    Process a specific split of the dataset and return annotations and images.
    """
    cap_annotations = []
    cap_images = []
    mask_annotations = []
    mask_images = []
    ann_index = ann_index_start
    cap_id = cap_id_start  # Start caption ID from the provided start value
    processed_frames = 0
    
    for image_index, frame_key in enumerate(tqdm(frame_keys, desc=f"Processing {dataset.name} {split_name}", unit="frame")):
        try:
            dataset.load_image(frame_key)

            # Load Caption file with unique ID
            single_cap_annotation, single_cap_image = generate_caption_entry(dataset, frame_key, cap_id, split_name)
            labels = single_cap_annotation['labels']
            
            cap_annotations.append(single_cap_annotation)
            cap_images.append(single_cap_image)

            # Load Mask file 
            single_mask_annotation, single_mask_image = generate_mask_entry(dataset, frame_key, ann_index, split_name)
            
            found_masks = len(single_mask_annotation) 
            mask_annotations.extend(single_mask_annotation)
            mask_images.append(single_mask_image)
            ann_index += found_masks
            cap_id += 1  # Increment caption ID for next frame
            assert len(labels) == len(single_mask_annotation)

            processed_frames += 1 

        except Exception as e:
            print(f"Error processing frame {frame_key}: {e}")
            continue
    
    return cap_annotations, cap_images, mask_annotations, mask_images, ann_index, cap_id, processed_frames

def copy_image_to_output(image_path, output_dir, original_filename, dataset_name):
    """
    Copy image to output directory maintaining folder structure.
    Returns the new relative path for the image.
    """
    # Create dataset-specific folder
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
            
    # Full output path
    output_path = os.path.join(dataset_output_dir, original_filename)
    
    # Copy the image
    shutil.copy2(image_path, output_path)
    
def get_base_dataset_name(dataset_name):
    """
    Extract base dataset name by removing _val, _train, _test suffixes.
    Returns lowercase version.
    """
    # Remove common suffixes
    suffixes_to_remove = ['_val', '_train', '_test']
    base_name = dataset_name
    
    for suffix in suffixes_to_remove:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break
    
    return base_name.lower()

def generate_caption_entry(dataset, frame_key, i, split_name="train"):
    """
    Generate caption annotation and image entry for a given frame_key.
    Raises exceptions on any error to stop the program.
    """
    # Validate inputs
    if not dataset:
        raise ValueError(f"Invalid dataset: {dataset}")
    if not frame_key:
        raise ValueError(f"Invalid frame_key: {frame_key}")
    if i is None:
        raise ValueError(f"Invalid index: {i}")
    
    # Check if frame_key exists in dataset
    if frame_key not in dataset.file_list:
        raise ValueError(f"frame_key '{frame_key}' not found in dataset")
    
    # caption annotation
    single_annotation = {}
    
    # Get caption and validate return
    caption = dataset.get_caption(frame_key)
    label_matched, caption_cleaned = parse_caption(caption)
    
    if caption is None:
        raise ValueError(f"get_caption returned None for frame_key: {frame_key}")
    if not caption or caption.strip() == "":
        raise ValueError(f"get_caption returned empty caption for frame_key: {frame_key}")
    single_annotation['caption'] = caption_cleaned
    single_annotation['caption_ann'] = caption
    single_annotation['label_matched'] = label_matched
    
    single_annotation['id'] = int(i)
    frame_key_no_ext = os.path.splitext(frame_key)[0]
    if dataset.is_video_dataset:
        frame_key_no_ext = "_".join(frame_key_no_ext.rsplit("/", 1))
    single_annotation['image_id'] = frame_key_no_ext
    
    # Get labels and validate return
    labels = dataset.get_labels(frame_key)
    if labels is None:
        raise ValueError(f"get_labels returned None for frame_key: {frame_key}")
    if not labels or len(labels) == 0:
        raise ValueError(f"get_labels returned empty list for frame_key: {frame_key}")
    single_annotation['labels'] = labels

    # caption image
    single_image = {}
    image_path, mask_path, metadata_key = dataset._get_paths_and_key(frame_key)

    # Copy image to output directory
    output_root = "processed_data/images"
    os.makedirs(output_root, exist_ok=True)
    
    # Determine folder name based on split (all lowercase)
    if split_name in ["test", "val"]:
        folder_name = "test_eval"
    else:
        # Get base dataset name (remove _val, _train, _test suffixes) and make lowercase
        folder_name = get_base_dataset_name(dataset.name)
    
    copy_image_to_output(image_path, output_root, frame_key_no_ext + ".jpg", folder_name)
    
    # Validate paths exist
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask path does not exist: {mask_path}")
    
    with Image.open(image_path) as img:
        width, height = img.size
    
    # Validate image dimensions
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height} for frame_key: {frame_key}")
    
    # Use folder name for file_name (this matches where the image is actually stored)
    single_image['file_name'] = folder_name + "/" + frame_key_no_ext + ".jpg"
    single_image['height'] = height
    single_image['id'] = frame_key_no_ext
    single_image['width'] = width

    # Final validation - check that all required fields are not empty
    if not single_annotation['caption']:
        raise ValueError(f"Empty caption in final annotation for frame_key: {frame_key}")
    if not single_annotation['labels']:
        raise ValueError(f"Empty labels in final annotation for frame_key: {frame_key}")
    if not single_image['file_name']:
        raise ValueError(f"Empty file_name in final image for frame_key: {frame_key}")
    if single_image['height'] <= 0 or single_image['width'] <= 0:
        raise ValueError(f"Invalid dimensions in final image for frame_key: {frame_key}")

    return single_annotation, single_image

def generate_mask_entry(dataset, frame_key, i, split_name="train"):
    """
    Generate mask annotation and image entry for a given frame_key.
    Raises exceptions on any error to stop the program.
    """
    # Validate inputs
    if not dataset:
        raise ValueError(f"Invalid dataset: {dataset}")
    if not frame_key:
        raise ValueError(f"Invalid frame_key: {frame_key}")
    if i is None:
        raise ValueError(f"Invalid index: {i}")
    
    # Check if frame_key exists in dataset
    if frame_key not in dataset.file_list:
        raise ValueError(f"frame_key '{frame_key}' not found in dataset")
    
    # masks annotation
    annotations_per_image = dataset.load_image_annotation(frame_key, i)

    # masks image
    single_image = {}
    image_path, mask_path, metadata_key = dataset._get_paths_and_key(frame_key)
    
    # Validate paths exist
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask path does not exist: {mask_path}")
    
    with Image.open(image_path) as img:
        width, height = img.size
    
    # Validate image dimensions
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height} for frame_key: {frame_key}")
    
    # Use frame_key as file_name directly (no extraction)
    frame_key_no_ext = os.path.splitext(frame_key)[0]
    if dataset.is_video_dataset:
        frame_key_no_ext = "_".join(frame_key_no_ext.rsplit("/", 1))
    
    # Determine folder name based on split (all lowercase)
    if split_name in ["test", "val"]:
        folder_name = "test_eval"
    else:
        # Get base dataset name (remove _val, _train, _test suffixes) and make lowercase
        folder_name = get_base_dataset_name(dataset.name)
    
    # Use folder name for file_name (this matches where the image is actually stored)
    single_image['file_name'] = folder_name + "/" + frame_key_no_ext + ".jpg"
    single_image['height'] = height
    single_image['id'] = frame_key_no_ext
    single_image['width'] = width

    # Final validation - check that all required fields are not empty
    if not single_image['file_name']:
        raise ValueError(f"Empty file_name in final image for frame_key: {frame_key}")
    if single_image['height'] <= 0 or single_image['width'] <= 0:
        raise ValueError(f"Invalid dimensions in final image for frame_key: {frame_key}")

    return annotations_per_image, single_image

def main():
    """Main function to process all annotated folders"""

    # Set random seed for reproducible splits
    random.seed(42)

    # Load caption folder 
    annotated_folders_path = "../annotated_folders/08_09"
    if not os.path.exists(annotated_folders_path):
        print(f"Error: {annotated_folders_path} does not exist")
        return
    print(f"Loading captiond from {annotated_folders_path}")

    # Load dataset configs
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

    # Initialize content dictionaries for combined test/val splits
    test_captions = {'annotations': [], 'images': []}
    val_captions = {'annotations': [], 'images': []}
    test_masks = {'annotations': [], 'images': [], 'categories': [{"id": 1, "name": "object"}]}
    val_masks = {'annotations': [], 'images': [], 'categories': [{"id": 1, "name": "object"}]}

    ann_index = 0

    # Process each subfolder
    for folder_name in subfolders:

        print(f"Processing subfolder: {folder_name}")

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

        try:
            # Create dataset
            dataset = CaptionPanopticDataset(
                resolved_name,
                dataset_config["image_dir"],
                dataset_config["ann_file"],
                dataset_config["mask_dir"],
                caption_dir=folder_path,
            )

            # Load dataset
            dataset.load()
            dataset.load_captions()

            # Get captioned frames
            all_frame_keys = dataset.file_list
            captioned_frame_keys = [frame_key for frame_key in all_frame_keys if dataset.has_caption_for_frame(frame_key)]
            print(f"Total frames: {len(all_frame_keys)}")
            print(f"Frames with captions: {len(captioned_frame_keys)}")

            # Check if this is a validation dataset that needs splitting
            is_val_dataset = 'val' in folder_name.lower()
            
            if is_val_dataset:
                # Split validation data 70/30 (test/val)
                test_keys, val_keys = split_validation_data(captioned_frame_keys)
                print(f"Split validation data: {len(test_keys)} test, {len(val_keys)} val")
                
                # Process test split with unique IDs
                test_cap_ann, test_cap_img, test_mask_ann, test_mask_img, test_ann_index, test_cap_id, test_processed = process_dataset_split(
                    dataset, test_keys, "test", len(test_masks['annotations']), len(test_captions['annotations'])
                )
                
                test_captions['annotations'].extend(test_cap_ann)
                test_captions['images'].extend(test_cap_img)
                test_masks['annotations'].extend(test_mask_ann)
                test_masks['images'].extend(test_mask_img)
                print(f"Test split: {test_processed}/{len(test_keys)} frames processed")
                
                # Process val split with unique IDs
                val_cap_ann, val_cap_img, val_mask_ann, val_mask_img, val_ann_index, val_cap_id, val_processed = process_dataset_split(
                    dataset, val_keys, "val", len(val_masks['annotations']), len(val_captions['annotations'])
                )
                val_captions['annotations'].extend(val_cap_ann)
                val_captions['images'].extend(val_cap_img)
                val_masks['annotations'].extend(val_mask_ann)
                val_masks['images'].extend(val_mask_img)
                print(f"Val split: {val_processed}/{len(val_keys)} frames processed")
                
                # Update global ann_index to the maximum of test and val
                ann_index = max(test_ann_index, val_ann_index)
                
            else:
                # Training data - process and save separately
                train_cap_ann, train_cap_img, train_mask_ann, train_mask_img, ann_index, train_cap_id, train_processed = process_dataset_split(
                    dataset, captioned_frame_keys, "train", ann_index
                )
                
                # Create individual training dataset content
                train_captions = {'annotations': train_cap_ann, 'images': train_cap_img}
                train_masks = {'annotations': train_mask_ann, 'images': train_mask_img, 'categories': [{"id": 1, "name": "object"}]}
                
                # Save individual training dataset
                try:
                    # Create annotations directory
                    os.makedirs("processed_data/annotations", exist_ok=True)
                    
                    with open(f"processed_data/annotations/{resolved_name}_caption.json", "w") as f:
                        json.dump(train_captions, f, indent=2)
                    print(f"Saved {resolved_name} train captions: {len(train_cap_ann)} annotations, {len(train_cap_img)} images")
                    
                    with open(f"processed_data/annotations/{resolved_name}_mask.json", "w") as f:
                        json.dump(train_masks, f, indent=2)
                    print(f"Saved {resolved_name} train masks: {len(train_mask_ann)} annotations, {len(train_mask_img)} images")
                    
                except Exception as e:
                    print(f"Error saving {resolved_name} train results: {e}")
                
                print(f"Train split: {train_processed}/{len(captioned_frame_keys)} frames processed")

        except Exception as e:
            print(f"Error processing dataset {folder_name}: {e}")
            continue
    
    # Save combined test and val results
    splits_to_save = [
        ('test', test_captions, test_masks),
        ('val', val_captions, val_masks)
    ]
    
    for split_name, cap_content, mask_content in splits_to_save:
        if cap_content['annotations']:  # Only save if there are annotations
            try:
                # Create annotations directory
                os.makedirs("processed_data/annotations", exist_ok=True)
                
                with open(f"processed_data/annotations/{split_name}_caption.json", "w") as f:
                    json.dump(cap_content, f, indent=2)
                print(f"Saved combined {split_name} captions: {len(cap_content['annotations'])} annotations, {len(cap_content['images'])} images")
                
                with open(f"processed_data/annotations/{split_name}_mask.json", "w") as f:
                    json.dump(mask_content, f, indent=2)
                print(f"Saved combined {split_name} masks: {len(mask_content['annotations'])} annotations, {len(mask_content['images'])} images")
                
            except Exception as e:
                print(f"Error saving combined {split_name} results: {e}")

if __name__ == "__main__":
    main()
    # WARNING IMAGES ARE SUPPOSED TO BE JPEG AND THE FOLDER TO START HAS TO BE CHANGED
