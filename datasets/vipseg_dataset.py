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
from collections import defaultdict, Counter
import random
from tqdm import tqdm
random.seed(42)
np.random.seed(42)

class VIPSegDataset(BaseDataset):
    def __init__(self, name, image_dir, ann_file, mask_dir):
        self.name = name
        self.image_dir = image_dir
        self.ann_file = ann_file
        self.mask_dir = mask_dir
        self.file_list = []
        self.labels = {}
        self.areas = {}
        self.coverages = {}
        self.video_map = {}  # fname -> video_id
        self.segments_info = {}  # fname -> original segments_info
        self._metadata_registered = False
        self.is_video_dataset = False # Auto-detected during load

        # Global stats to be computed once
        self.all_labels = []
        self.goal_freqs = []
        self.goal_areas = []
        self.goal_mask_counts = []
        self.goal_unique_labels = []

        # Running stats for selected files
        self.current_label_counter = Counter()
        self.current_area_counter = Counter()
        self.selected_files = set()

    def load(self):
        print(f"Loading {self.name} dataset... This may take a few seconds.")

        with open(self.ann_file, 'r') as f:
            data = json.load(f)

        if not self._metadata_registered:
            self._register_metadata(data["categories"])
            self._metadata_registered = True

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

        all_labels_set = set()
        label_counter = Counter()
        area_counter = Counter()
        mask_counts = []
        unique_label_counts = []
        processed_items = set()

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


    def _register_metadata(self, categories):
        dataset_name = self.name
        if dataset_name in MetadataCatalog.list():
            return

        thing_cats = [c for c in categories if c.get("isthing", 0)]
        stuff_cats = [c for c in categories if not c.get("isthing", 0)]

        meta = MetadataCatalog.get(dataset_name)
        meta.thing_classes = [c['name'] for c in thing_cats]
        meta.stuff_classes = [c['name'] for c in stuff_cats]
        meta.thing_dataset_id_to_contiguous_id = {c['id']: i for i, c in enumerate(thing_cats)}
        meta.stuff_dataset_id_to_contiguous_id = {c['id']: i for i, c in enumerate(stuff_cats)}

    def _get_label_name(self, cat_id):
        meta = MetadataCatalog.get(self.name)
        if cat_id in meta.thing_dataset_id_to_contiguous_id:
            idx = meta.thing_dataset_id_to_contiguous_id[cat_id]
            return meta.thing_classes[idx]
        elif cat_id in meta.stuff_dataset_id_to_contiguous_id:
            idx = meta.stuff_dataset_id_to_contiguous_id[cat_id]
            return meta.stuff_classes[idx]
        return str(cat_id)

    def load_image(self, fname):
        base_name, _ = os.path.splitext(fname)

        if self.is_video_dataset:
            video_id = self.video_map.get(fname)
            image_path = os.path.join(self.image_dir, video_id, f"{base_name}.jpg")
            mask_path = os.path.join(self.mask_dir, video_id, f"{base_name}.png")
        else:
            image_path = os.path.join(self.image_dir, f"{base_name}.jpg")
            mask_path = os.path.join(self.mask_dir, f"{base_name}.png")

        image = Image.open(image_path).convert("RGB")
        mask = np.array(Image.open(mask_path))
        panoptic_seg = rgb2id(mask).astype(np.int32)

        # Create mapping from panoptic IDs to contiguous category IDs
        meta = MetadataCatalog.get(self.name)
        thing_map = meta.thing_dataset_id_to_contiguous_id
        stuff_map = meta.stuff_dataset_id_to_contiguous_id

        segments_info = self.segments_info.get(fname, [])
        segments_info_mapped = []

        for seg in segments_info:
            new_seg = seg.copy()
            cat_id = seg["category_id"]
            if cat_id in thing_map:
                new_seg["isthing"] = True
                new_seg["category_id"] = thing_map[cat_id]
            elif cat_id in stuff_map:
                new_seg["isthing"] = False
                new_seg["category_id"] = stuff_map[cat_id]
            else:
                new_seg["isthing"] = False
                # Optional: skip unknown categories
            segments_info_mapped.append(new_seg)


        visualizer = Visualizer(np.array(image), meta, instance_mode=ColorMode.IMAGE)
        vis_output = visualizer.draw_panoptic_seg_predictions(panoptic_seg=torch.from_numpy(panoptic_seg), segments_info=segments_info_mapped)
        vis_img = vis_output.get_image()
        qimage = QImage(vis_img.data, vis_img.shape[1], vis_img.shape[0], vis_img.strides[0], QImage.Format.Format_RGB888)

        # Enhanced label formatting
        label_names = [f"{i}: {self._get_label_name(label)}" for i, label in enumerate(self.labels[fname])]
        coverage = self.coverages.get(fname, 0)

        # Append coverage to label output
        label_names.append(f"\nCoverage: {coverage:.2f}%")

        return QImage(image_path), qimage, label_names, f"Coverage: {coverage:.2f}%"

    def get_single_segment_visualization(self, fname, segment_index):
        """
        Generates a QImage visualizing only a single panoptic segment.
        """
        base_name, _ = os.path.splitext(fname)

        # 1. Load original image and panoptic segmentation mask
        if self.is_video_dataset:
            video_id = self.video_map.get(fname)
            image_path = os.path.join(self.image_dir, video_id, f"{base_name}.jpg")
            mask_path = os.path.join(self.mask_dir, video_id, f"{base_name}.png")
        else:
            image_path = os.path.join(self.image_dir, f"{base_name}.jpg")
            mask_path = os.path.join(self.mask_dir, f"{base_name}.png")

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        panoptic_seg = torch.from_numpy(rgb2id(mask).astype(np.int32))

        # 2. Get all segments for this image, but select only one
        all_segments_info = self.segments_info.get(fname, [])
        if not (0 <= segment_index < len(all_segments_info)):
            return None  # Invalid index

        segment_to_show = all_segments_info[segment_index]

        # 3. Map the category ID for the visualizer
        meta = MetadataCatalog.get(self.name)
        thing_map = meta.thing_dataset_id_to_contiguous_id
        stuff_map = meta.stuff_dataset_id_to_contiguous_id

        new_seg = segment_to_show.copy()
        cat_id = new_seg["category_id"]
        if cat_id in thing_map:
            new_seg["isthing"] = True
            new_seg["category_id"] = thing_map[cat_id]
        elif cat_id in stuff_map:
            new_seg["isthing"] = False
            new_seg["category_id"] = stuff_map[cat_id]

        # 4. Create the visualization with only the single segment
        visualizer = Visualizer(image, meta, instance_mode=ColorMode.IMAGE)
        vis_output = visualizer.draw_panoptic_seg_predictions(panoptic_seg=panoptic_seg, segments_info=[new_seg])
        vis_img = vis_output.get_image()
        return QImage(vis_img.data, vis_img.shape[1], vis_img.shape[0], vis_img.strides[0], QImage.Format.Format_RGB888)

    def get_goal_stats(self):
        return self.all_labels, self.goal_freqs, self.goal_areas

    def get_current_stats(self, selected_files):
        label_counter = Counter()
        area_counter = Counter()

        for fname in selected_files:
            for label in self.labels.get(fname, []):
                label_counter[label] += 1
            for label, area in self.areas.get(fname, {}).items():
                area_counter[label] += area

        return self.all_labels, [label_counter[label] for label in self.all_labels], [area_counter[label] for label in self.all_labels]


    def get_selected_histograms(self, selected_files):
        mask_counts = []
        unique_label_counts = []
        for fname in selected_files:
            mask_counts.append(len(self.segments_info.get(fname, [])))
            unique_label_counts.append(len(set(self.labels.get(fname, []))))
        return mask_counts, unique_label_counts

    def get_goal_histograms(self):
        return self.goal_mask_counts, self.goal_unique_labels