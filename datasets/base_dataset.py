from abc import ABC, abstractmethod
from typing import List, Dict
from collections import Counter
from PyQt6.QtGui import QImage


class BaseDataset(ABC):
    def __init__(self, name: str):
        self.name = name
        self.file_list: List[str] = []
        self.images: Dict[str, QImage] = {}
        self.masks: Dict[str, QImage] = {}
        self.labels: Dict[str, List[int]] = {}
        self.areas: Dict[str, Dict[int, float]] = {}
        self.coverages: Dict[str, float] = {}
        self.segments_info: Dict[str, List[Dict]] = {}

        self.all_labels = []
        self.goal_freqs = []
        self.goal_areas = []
        self.goal_mask_counts = []
        self.goal_unique_labels = []

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def get_goal_stats(self):
        pass

    def get_current_stats(self, selected_files: List[str]):
        label_counter = Counter()
        area_counter = Counter()

        for fname in selected_files:
            for label in self.labels.get(fname, []):
                label_counter[label] += 1
            for label, area in self.areas.get(fname, {}).items():
                area_counter[label] += area

        return self.all_labels, [label_counter[label] for label in self.all_labels], [area_counter[label] for label in self.all_labels]

    def get_selected_histograms(self, selected_files: List[str]):
        mask_counts = []
        unique_label_counts = []
        for fname in selected_files:
            mask_counts.append(len(self.segments_info.get(fname, [])))
            unique_label_counts.append(len(set(self.labels.get(fname, []))))
        return mask_counts, unique_label_counts

    def get_goal_histograms(self):
        return self.goal_mask_counts, self.goal_unique_labels

    def get_goal_stats(self):
        return self.all_labels, self.goal_freqs, self.goal_areas