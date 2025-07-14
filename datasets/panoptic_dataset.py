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

class PanopticDataset(BaseDataset):
    def __init__(self, name, image_dir, ann_file, mask_dir):
        super().__init__(name)
        self.image_dir = image_dir
        self.ann_file = ann_file
        self.mask_dir = mask_dir
        self.is_video_dataset = False
        self.visualizer_segments = {}

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

        id_to_label = []  # Indexed by segment order
        viz_segments = []  # Segments with category_id remapped for Visualizer

        thing_classes = []
        stuff_classes = []
        thing_idx = 0
        stuff_idx = 0

        for seg in segments_info:
            cat_id = seg["category_id"]
            name = self.categories[cat_id]["name"]
            is_thing = self.category_id_isthing.get(cat_id, 0) == 1

            seg_copy = seg.copy()
            seg_copy["isthing"] = is_thing

            # Create the label string with the correct global index for both UI and visualization
            label_str = f"{len(id_to_label)}: {name}"
            id_to_label.append(label_str)

            # Remap category_id for Visualizer
            if is_thing:
                seg_copy["category_id"] = thing_idx
                thing_classes.append(label_str)
                thing_idx += 1
            else:
                seg_copy["category_id"] = stuff_idx
                stuff_classes.append(label_str)
                stuff_idx += 1

            viz_segments.append(seg_copy)

        if metadata_key not in MetadataCatalog.list():
            meta = MetadataCatalog.get(metadata_key)
            meta.thing_classes = thing_classes
            meta.stuff_classes = stuff_classes

        visualizer = Visualizer(np.array(image), MetadataCatalog.get(metadata_key), instance_mode=ColorMode.IMAGE)
        vis_output = visualizer.draw_panoptic_seg_predictions(
            panoptic_seg=torch.from_numpy(panoptic_seg),
            segments_info=viz_segments
        )
        self.visualizer_segments[frame_key] = viz_segments
        vis_img = vis_output.get_image()
        qimage = QImage(vis_img.data, vis_img.shape[1], vis_img.shape[0], vis_img.strides[0], QImage.Format.Format_RGB888)

        coverage = self.coverages.get(frame_key, 0.0)
        id_to_label.append(f"\nCoverage: {coverage:.2f}%")

        return QImage(image_path), qimage, id_to_label, f"Coverage: {coverage:.2f}%"

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

        if metadata_key not in MetadataCatalog.list():
            raise KeyError(f"Metadata '{metadata_key}' not registered. Call load_image() first.")

        visualizer = Visualizer(image, MetadataCatalog.get(metadata_key), instance_mode=ColorMode.IMAGE)
        vis_output = visualizer.draw_panoptic_seg_predictions(
            panoptic_seg=panoptic_seg,
            segments_info=[vis_segments[segment_index]]
        )
        vis_img = vis_output.get_image()

        return QImage(
            vis_img.data,
            vis_img.shape[1],
            vis_img.shape[0],
            vis_img.strides[0],
            QImage.Format.Format_RGB888
        )
