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
        self.video_map = {}
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
                    video_id = video['video_id']
                    for frame in video.get('annotations', []):
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
            if self.is_video_dataset:
                unique_id = (frame['video_id'], fname)
            else:
                unique_id = fname

            if unique_id in processed_items:
                # Avoid processing duplicate entries
                skipped_duplicates += 1
                continue

            processed_items.add(unique_id)
            base_name, _ = os.path.splitext(fname)

            # Construct paths based on dataset type
            if self.is_video_dataset:
                video_id = frame['video_id']
                image_path = os.path.join(self.image_dir, video_id, f"{base_name}.jpg")
                mask_path = os.path.join(self.mask_dir, video_id, f"{base_name}.png")
            else:
                video_id = None  # No video_id for flat datasets
                image_path = os.path.join(self.image_dir, f"{base_name}.jpg")
                mask_path = os.path.join(self.mask_dir, f"{base_name}.png")

            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                skipped_missing_files += 1
                continue

            segments_info = frame.get('segments_info', [])
            self.segments_info[fname] = segments_info

            labels = [seg['category_id'] for seg in segments_info]
            area_map = {seg['category_id']: seg.get('area', 0) for seg in segments_info}

            try:
                panoptic_seg = np.array(Image.open(mask_path))
                panoptic_seg = rgb2id(panoptic_seg).astype(np.int32)
                labeled_pixels = np.sum(panoptic_seg != 0)
                total_pixels = panoptic_seg.shape[0] * panoptic_seg.shape[1]
                coverage = (labeled_pixels / total_pixels) * 100

                self.file_list.append(fname)
                self.labels[fname] = labels
                self.areas[fname] = area_map
                self.coverages[fname] = coverage
                if video_id:
                    self.video_map[fname] = video_id

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

    def _get_paths_and_key(self, fname):
        """Helper to construct paths and metadata key for a given filename."""
        base_name, _ = os.path.splitext(fname)

        if self.is_video_dataset:
            video_id = self.video_map.get(fname)
            if not video_id:
                raise ValueError(f"Video ID not found for '{fname}'")
            image_path = os.path.join(self.image_dir, video_id, f"{base_name}.jpg")
            mask_path = os.path.join(self.mask_dir, video_id, f"{base_name}.png")
            metadata_key = f"{self.name}_{video_id}_{base_name}"
        else:
            image_path = os.path.join(self.image_dir, f"{base_name}.jpg")
            mask_path = os.path.join(self.mask_dir, f"{base_name}.png")
            metadata_key = f"{self.name}_{base_name}"

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        return image_path, mask_path, metadata_key

    def load_image(self, fname):
        image_path, mask_path, metadata_key = self._get_paths_and_key(fname)

        image = Image.open(image_path).convert("RGB")
        mask = np.array(Image.open(mask_path))
        panoptic_seg = rgb2id(mask).astype(np.int32)

        segments_info = self.segments_info.get(fname)
        if segments_info is None:
            raise ValueError(f"No segments found for {fname}")

        id_to_label = []  # Indexed by segment order
        viz_segments = []  # Segments with category_id remapped for Visualizer

        thing_classes = []
        stuff_classes = []
        thing_id_map = {}
        stuff_id_map = {}
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
        self.visualizer_segments[fname] = viz_segments
        vis_img = vis_output.get_image()
        qimage = QImage(vis_img.data, vis_img.shape[1], vis_img.shape[0], vis_img.strides[0], QImage.Format.Format_RGB888)

        coverage = self.coverages.get(fname, 0.0)
        id_to_label.append(f"\nCoverage: {coverage:.2f}%")

        return QImage(image_path), qimage, id_to_label, f"Coverage: {coverage:.2f}%"

    def get_single_segment_visualization(self, fname, segment_index):
        """
        Visualizes a single panoptic segment using per-image metadata.
        Assumes load_image(fname) was called beforehand to register metadata.
        """
        image_path, mask_path, metadata_key = self._get_paths_and_key(fname)

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        panoptic_seg = torch.from_numpy(rgb2id(mask).astype(np.int32))

        vis_segments = self.visualizer_segments.get(fname)
        if vis_segments is None:
            raise RuntimeError(f"Visualizer segments not cached for {fname}. Call load_image() first.")

        if not (0 <= segment_index < len(vis_segments)):
            raise IndexError(f"Invalid segment index {segment_index} for '{fname}'")

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
