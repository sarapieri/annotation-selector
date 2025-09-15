import os
import json
import numpy as np
from PIL import Image
from panopticapi.utils import rgb2id
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from PyQt6.QtGui import QImage
from datasets.base_dataset import BaseDataset
import torch
from collections import Counter
import random
from tqdm import tqdm
random.seed(42)
np.random.seed(42)
from pycocotools import mask as mask_util

class PanopticDataset(BaseDataset):
    def __init__(self, name, image_dir, ann_file, mask_dir):
        super().__init__(name)
        self.image_dir = image_dir
        self.ann_file = ann_file
        self.mask_dir = mask_dir
        self.caption_dir = None  # Optional caption directory
        self.is_video_dataset = False
        self.visualizer_segments = {}
        self.font_size = 25 if "VIPSeg" in name else 10
        self.captions = {}  # frame_key -> caption text
        self.visualization_labels = {}  # frame_key -> list of label strings

    def load(self):
        print(f"Loading {self.name} dataset... This may take a few seconds.")

        try:
            with open(self.ann_file, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error: Failed to load or parse annotation file '{self.ann_file}'. {e}")
            return # Stop loading if the main annotation file is invalid

        # Auto-detect dataset type and prepare a flat list of annotations
        annotations_list = []
        # Check if 'annotations' key exists and is not empty
        if data.get('annotations') and isinstance(data['annotations'], list) and data['annotations']:
            # Check if the first annotation has a 'video_id', suggesting a video dataset
            if 'video_id' in data['annotations'][0]:
                self.is_video_dataset = True
                print("Detected video dataset format.")
                for video in data['annotations']:
                    # Defensive checks for each video entry
                    if 'video_id' not in video:
                        print(f"Warning: Skipping an entry in annotations list because 'video_id' is missing.")
                        continue
                    if 'annotations' not in video or not video['annotations']:
                        print(f"Warning: No frames found or 'annotations' key missing for video_id: {video['video_id']}. Skipping.")
                        continue

                    video_id = video['video_id']
                    for frame in video['annotations']:
                        frame['video_id'] = video_id  # Inject video_id for unified processing
                        annotations_list.append(frame)
            else: # Assumed to be an image dataset
                self.is_video_dataset = False
                print("Detected image dataset format.")
                annotations_list = data.get('annotations', [])

        # === Load categories and category mapping ===
        categories = data.get("categories")
        if not categories:
            raise ValueError(f"No 'categories' found in annotation file '{self.ann_file}'")

        # Map category ID to full category dict
        self.categories = {cat["id"]: cat for cat in categories}
        # Map category ID to isthing boolean
        self.category_id_isthing = {cat["id"]: cat.get("isthing", 0) for cat in categories}

        all_labels_set = set()
        label_counter = Counter()
        area_counter = Counter()
        mask_counts = []
        unique_label_counts = []
        processed_items = set()
        skipped_duplicates = 0
        skipped_missing_files = 0
        empty_mask_files = 0
        segments_without_pixels = 0
        total_segments_checked = 0

        for frame in tqdm(annotations_list, desc=f"Processing {self.name}"):
            fname = frame['file_name']
            # Create a unique key for each frame to handle cases where file names
            # are repeated across different videos.
            if self.is_video_dataset:
                video_id = frame['video_id']
                frame_key = f"{video_id}/{fname}"
            else:
                video_id = None
                frame_key = fname

            if frame_key in processed_items:
                # Avoid processing duplicate entries from the annotation file
                skipped_duplicates += 1
                continue

            processed_items.add(frame_key)
            base_name, _ = os.path.splitext(fname)

            # Construct paths based on dataset type
            if self.is_video_dataset:
                image_path = os.path.join(self.image_dir, video_id, f"{base_name}.jpg")
                mask_path = os.path.join(self.mask_dir, video_id, f"{base_name}.png")
            else:
                image_path = os.path.join(self.image_dir, f"{base_name}.jpg")
                mask_path = os.path.join(self.mask_dir, f"{base_name}.png")

            # Check for file existence and provide specific feedback for debugging
            image_exists = os.path.exists(image_path)
            mask_exists = os.path.exists(mask_path)
            if not image_exists or not mask_exists:
                if not image_exists:
                    print(f"Warning: Image file not found, skipping frame. Path: {image_path}")
                if not mask_exists:
                    print(f"Warning: Mask file not found, skipping frame. Path: {mask_path}")
                skipped_missing_files += 1
                continue

            segments_info = frame.get('segments_info', [])
            if not segments_info:
                print(f"Warning: Frame '{frame_key}' has no 'segments_info'. It will be processed but may appear empty.")

            labels = [seg['category_id'] for seg in segments_info]
            area_map = {seg['category_id']: seg.get('area', 0) for seg in segments_info}

            try:
                panoptic_seg = np.array(Image.open(mask_path))
                panoptic_seg = rgb2id(panoptic_seg).astype(np.int32)
                labeled_pixels = np.sum(panoptic_seg != 0)
                total_pixels = panoptic_seg.shape[0] * panoptic_seg.shape[1]
                coverage = (labeled_pixels / total_pixels) * 100

                # Check for completely empty masks
                if labeled_pixels == 0:
                    empty_mask_files += 1
                    print(f"Warning: Frame '{frame_key}' has an empty mask (no labeled pixels)")

                # Check which segments actually exist in the mask
                existing_ids = set(np.unique(panoptic_seg))
                if 0 in existing_ids:  # Remove background ID if present
                    existing_ids.remove(0)
                
                # Count segments that don't exist in the mask
                segments_in_annotation = set(seg['id'] for seg in segments_info)
                segments_not_in_mask = segments_in_annotation - existing_ids
                if segments_not_in_mask:
                    segments_without_pixels += len(segments_not_in_mask)
                    print(f"Warning: Frame '{frame_key}' has {len(segments_not_in_mask)} segments in annotation but not in mask: {segments_not_in_mask}")
                
                total_segments_checked += len(segments_info)

                self.segments_info[frame_key] = segments_info
                self.file_list.append(frame_key)
                self.labels[frame_key] = labels
                self.areas[frame_key] = area_map
                self.coverages[frame_key] = coverage

                all_labels_set.update(labels)
                label_counter.update(labels)
                area_counter.update(area_map)
                mask_counts.append(len(segments_info))
                unique_label_counts.append(len(set(labels)))
            except Exception as e:
                print(f"Warning: Could not process mask file {mask_path}. Error: {e}")
                continue

        self.all_labels = sorted(list(all_labels_set))
        self.goal_freqs = [label_counter[label] for label in self.all_labels]
        self.goal_areas = [area_counter[label] for label in self.all_labels]
        self.goal_mask_counts = mask_counts
        self.goal_unique_labels = unique_label_counts

        print(f"{self.name} dataset loaded: {len(self.file_list)} files processed.")
        if skipped_duplicates > 0:
            print(f"Skipped {skipped_duplicates} duplicate entries.")
        if skipped_missing_files > 0:
            print(f"Warning: Skipped {skipped_missing_files} entries due to missing image or mask files.")
        if empty_mask_files > 0:
            print(f"Warning: Found {empty_mask_files} frames with completely empty masks (no labeled pixels).")
        if segments_without_pixels > 0:
            print(f"Warning: Found {segments_without_pixels} segments in annotations that don't exist in their corresponding masks.")
        if total_segments_checked > 0:
            print(f"Mask quality: {segments_without_pixels}/{total_segments_checked} segments ({segments_without_pixels/total_segments_checked*100:.1f}%) have annotation-mask mismatches.")

    def set_caption_dir(self, caption_dir):
        """Set the caption directory and optionally load captions"""
        self.caption_dir = caption_dir
        if caption_dir and os.path.exists(caption_dir):
            self._load_captions()

    def _load_captions(self):
        """Load captions from the caption directory if available"""
        if not self.caption_dir or not os.path.exists(self.caption_dir):
            return

        print(f"Loading captions from {self.caption_dir}...")
        json_files = [f for f in os.listdir(self.caption_dir) if f.endswith('.json')]
        loaded_count = 0

        for filename in json_files:
            file_path = os.path.join(self.caption_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'text' not in data:
                    continue
                
                caption_value = data['text']
                if not isinstance(caption_value, str):
                    continue

                image_id = os.path.splitext(filename)[0]
                matched_key = None

                if self.is_video_dataset:
                    # Parse video caption filename: <video_id>_<frame_id>.json
                    video_id, sep, frame_base = image_id.rpartition('_')
                    if sep == '' or not frame_base:
                        continue
                    
                    # Look for exact match in file_list
                    for frame_key in self.file_list:
                        if frame_key == f"{video_id}/{frame_base}.png" or frame_key == f"{video_id}/{frame_base}.jpg":
                            matched_key = frame_key
                            break
                else:
                    # Image dataset: try numeric conversion first
                    try:
                        base_name = f"{int(image_id):012d}"
                        for frame_key in self.file_list:
                            if frame_key == f"{base_name}.jpg" or frame_key == f"{base_name}.png":
                                matched_key = frame_key
                                break
                    except ValueError:
                        # Non-numeric ID, try direct match
                        for frame_key in self.file_list:
                            if frame_key == f"{image_id}.jpg" or frame_key == f"{image_id}.png":
                                matched_key = frame_key
                                break

                if matched_key and matched_key not in self.captions:
                    self.captions[matched_key] = caption_value
                    loaded_count += 1

            except Exception as e:
                continue  # Skip files with errors

        print(f"Loaded {loaded_count} captions for {self.name}")

    def get_caption(self, frame_key):
        """Get caption for a specific frame if available"""
        return self.captions.get(frame_key, "")

    def has_caption_for_frame(self, frame_key):
        """Check if a frame has an associated caption"""
        return frame_key in self.captions

    def get_all_captions(self):
        """Get all captions"""
        return self.captions.copy()

    def update_caption_cache(self, frame_key, new_caption):
        """Update the caption cache with a new caption value."""
        self.captions[frame_key] = new_caption

    def get_caption_file_path(self, frame_key):
        """Get the file path for a caption file."""
        if not self.caption_dir:
            return None
            
        if self.is_video_dataset:
            # For video datasets, frame_key is "video_id/fname.ext"
            video_id, fname = frame_key.split('/', 1)
            base_name, _ = os.path.splitext(fname)
            caption_filename = f"{video_id}_{base_name}.json"
        else:
            # For image datasets, frame_key is "fname.ext"
            base_name, _ = os.path.splitext(frame_key)
            
            # Remove leading zeros for numeric IDs to match the loading pattern
            try:
                # Try to convert to int and back to remove leading zeros
                numeric_id = int(base_name)
                caption_filename = f"{numeric_id}.json"
            except ValueError:
                # If not numeric, use as-is
                caption_filename = f"{base_name}.json"
            
        return os.path.join(self.caption_dir, caption_filename)

    def save_caption(self, frame_key, new_caption):
        """Save a caption to the caption directory."""
        if not self.caption_dir:
            raise ValueError("No caption directory configured")
            
        caption_file_path = self.get_caption_file_path(frame_key)
        if not caption_file_path:
            raise ValueError("Could not determine caption file path")
            
        # Ensure the caption directory exists (create if it doesn't exist)
        os.makedirs(os.path.dirname(caption_file_path), exist_ok=True)
        
        # Save the new caption in JSON format
        caption_data = {"text": new_caption}
        with open(caption_file_path, 'w', encoding='utf-8') as f:
            json.dump(caption_data, f, indent=2, ensure_ascii=False)
            
        # Update the cache
        self.update_caption_cache(frame_key, new_caption)

    def _get_label_name(self, cat_id):  
        cat = self.categories.get(cat_id)
        return f"{cat_id}: {cat['name']}" if cat else str(cat_id)

    def _get_paths_and_key(self, frame_key):
        """Helper to construct paths and metadata key for a given filename."""

        if self.is_video_dataset:
            # For video datasets, frame_key is "video_id/fname.ext"
            video_id, fname = frame_key.split('/', 1)
            base_name, _ = os.path.splitext(fname)
            image_path = os.path.join(self.image_dir, video_id, f"{base_name}.jpg")
            mask_path = os.path.join(self.mask_dir, video_id, f"{base_name}.png")
            metadata_key = f"{self.name}_{video_id}_{base_name}"
        else:
            # For image datasets, frame_key is "fname.ext"
            fname = frame_key
            base_name, _ = os.path.splitext(fname)
            image_path = os.path.join(self.image_dir, f"{base_name}.jpg")
            mask_path = os.path.join(self.mask_dir, f"{base_name}.png")
            metadata_key = f"{self.name}_{base_name}"

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        return image_path, mask_path, metadata_key

    def load_image(self, frame_key):
        image_path, mask_path, metadata_key = self._get_paths_and_key(frame_key)

        image = Image.open(image_path).convert("RGB")
        mask = np.array(Image.open(mask_path))
        panoptic_seg = rgb2id(mask).astype(np.int32)

        segments_info = self.segments_info.get(frame_key)
        if segments_info is None:
            raise ValueError(f"No segments found for {frame_key}")

        # Get the set of segment IDs that actually exist in the mask
        existing_ids = set(np.unique(panoptic_seg))
        if 0 in existing_ids:  # Remove background ID if present
            existing_ids.remove(0)

        # Filter segments_info to only include segments that exist in the mask
        filtered_segments = []
        id_to_label = []
        thing_classes = []
        stuff_classes = []
        thing_idx = 0
        stuff_idx = 0
        label_list = []

        for seg in segments_info:
            if seg['id'] not in existing_ids:
                continue  # Skip segments that don't exist in the mask

            cat_id = seg["category_id"]
            name = self.categories[cat_id]["name"]
            is_thing = self.category_id_isthing.get(cat_id, 0) == 1

            seg_copy = seg.copy()
            seg_copy["isthing"] = is_thing

            # Create the label string with the correct global index
            label_str = f"{len(id_to_label)}"
            id_to_label.append(f"{len(id_to_label)}: {name}")
            label_list.append(name)

            # Remap category_id for Visualizer
            if is_thing:
                seg_copy["category_id"] = thing_idx
                thing_classes.append(label_str)
                thing_idx += 1
            else:
                seg_copy["category_id"] = stuff_idx
                stuff_classes.append(label_str)
                stuff_idx += 1

            filtered_segments.append(seg_copy)

        if metadata_key not in MetadataCatalog.list():
            meta = MetadataCatalog.get(metadata_key)
            meta.thing_classes = thing_classes
            meta.stuff_classes = stuff_classes

        visualizer = Visualizer(np.array(image), MetadataCatalog.get(metadata_key), instance_mode=ColorMode.IMAGE)
        visualizer._default_font_size = self.font_size
        vis_output = visualizer.draw_panoptic_seg_predictions(
            panoptic_seg=torch.from_numpy(panoptic_seg),
            segments_info=filtered_segments
        )
        self.visualizer_segments[frame_key] = filtered_segments
        self.visualization_labels[frame_key] = label_list
        vis_img = vis_output.get_image()
        qimage = QImage(vis_img.data, vis_img.shape[1], vis_img.shape[0], vis_img.strides[0], QImage.Format.Format_RGB888)

        return QImage(image_path), qimage, id_to_label
    
    def load_image_annotation(self, frame_key, start_annotation_id):
        image_path, mask_path, metadata_key = self._get_paths_and_key(frame_key)

        image = Image.open(image_path).convert("RGB")
        mask = np.array(Image.open(mask_path))
        panoptic_seg = rgb2id(mask).astype(np.int32)

        segments_info = self.segments_info.get(frame_key)
        if segments_info is None:
            raise ValueError(f"No segments found for {frame_key}")

        # Get the set of segment IDs that actually exist in the mask
        existing_ids = set(np.unique(panoptic_seg))
        if 0 in existing_ids:  # Remove background ID if present
            existing_ids.remove(0)

        # Filter segments_info to only include segments that exist in the mask
        annotations = []
        annotation_id = start_annotation_id
        
        for seg in segments_info:
            if seg['id'] not in existing_ids:
                continue  # Skip segments that don't exist in the mask

            cat_id = seg["category_id"]

            # Create binary mask for this segment
            segment_mask = (panoptic_seg == seg['id']).astype(np.uint8)
            
            # Check if mask has any pixels
            if np.sum(segment_mask) == 0:
                continue  # Skip empty masks
            
            try:
                # Convert to RLE format
                rle = mask_util.encode(np.asfortranarray(segment_mask))
                rle['counts'] = rle['counts'].decode('utf-8')
                
                # Calculate bounding box
                bbox = mask_util.toBbox(rle).tolist()
                # Calculate area
                area = int(mask_util.area(rle))


                frame_key_no_ext = os.path.splitext(frame_key)[0]
                if self.is_video_dataset:
                    frame_key_no_ext = "_".join(frame_key_no_ext.rsplit("/", 1))
                
                # Create annotation in COCO format
                annotation = {
                    'id': annotation_id,  # Unique ID for this annotation
                    'image_id': frame_key_no_ext,
                    'category_id': cat_id,  # Original category ID
                    'segmentation': {
                        'size': rle['size'],
                        'counts': rle['counts']
                    },
                    'area': area,
                    'bbox': bbox,
                    'iscrowd': 0
                }
                annotations.append(annotation)
                annotation_id += 1  # Increment for next segment
                
            except Exception as e:
                print(f"Warning: Failed to convert segment {seg['id']} to RLE: {e}")
                continue
        
        return annotations

    def get_single_segment_visualization(self, frame_key, segment_index):
        """
        Visualizes a single panoptic segment using per-image metadata.
        Assumes load_image(frame_key) was called beforehand to register metadata.
        """
        image_path, mask_path, metadata_key = self._get_paths_and_key(frame_key)

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        panoptic_seg = torch.from_numpy(rgb2id(mask).astype(np.int32))

        vis_segments = self.visualizer_segments.get(frame_key)
        if vis_segments is None:
            raise RuntimeError(f"Visualizer segments not cached for {frame_key}. Call load_image() first.")

        if not (0 <= segment_index < len(vis_segments)):
            raise IndexError(f"Invalid segment index {segment_index} for '{frame_key}'")

        target_segment = vis_segments[segment_index]
        segment_id = target_segment['id']

        # Check if segment exists in mask
        if segment_id not in set(torch.unique(panoptic_seg).tolist()):
            print(f"Warning: Segment ID {segment_id} not found in mask")
            print(f"Available segment IDs in mask: {torch.unique(panoptic_seg).tolist()}")

        if metadata_key not in MetadataCatalog.list():
            raise KeyError(f"Metadata '{metadata_key}' not registered. Call load_image() first.")

        visualizer = Visualizer(image, MetadataCatalog.get(metadata_key), instance_mode=ColorMode.IMAGE)
        visualizer._default_font_size = self.font_size
        vis_output = visualizer.draw_panoptic_seg_predictions(
            panoptic_seg=panoptic_seg,
            segments_info=[target_segment]
        )
        vis_img = vis_output.get_image()

        return QImage(
            vis_img.data,
            vis_img.shape[1],
            vis_img.shape[0],
            vis_img.strides[0],
            QImage.Format.Format_RGB888
        )

    def get_labels(self, frame_key):
        """Get the text labels that are shown in the visualization"""
        return self.visualization_labels.get(frame_key, [])
