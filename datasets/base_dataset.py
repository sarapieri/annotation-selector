from abc import ABC, abstractmethod
from typing import List, Dict
from PyQt6.QtGui import QImage


class BaseDataset(ABC):
    def __init__(self, name: str):
        self.name = name
        self.file_list: List[str] = []
        self.images: Dict[str, QImage] = {}
        self.masks: Dict[str, QImage] = {}
        self.labels: Dict[str, List[str]] = {}

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def get_goal_stats(self):
        pass

    @abstractmethod
    def get_current_stats(self, selected_files: List[str]):
        pass
