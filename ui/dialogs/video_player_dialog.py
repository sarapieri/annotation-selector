import os
from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from datasets.panoptic_dataset import PanopticDataset


class VideoPlayerDialog(QDialog):
    def __init__(self, dataset: PanopticDataset, video_id: str):
        super().__init__()
        self.setWindowTitle(f"Playing Video: {video_id}")
        self.setMinimumSize(600, 400)
        self.dataset = dataset
        self.video_id = video_id
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)

        # Get frames belonging to this video
        self.frames = [
            fname for fname, vid in dataset.video_map.items()
            if vid == video_id and os.path.exists(os.path.join(dataset.image_dir, vid, fname.replace(".png", ".jpg")))
        ]
        self.frames.sort()  # ensure ordered playback
        self.index = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        fps = 5
        self.timer.start(1000 // fps)

    def next_frame(self):
        if not self.frames:
            self.timer.stop()
            return
        fname = self.frames[self.index]
        video_id = self.video_id
        path = os.path.join(self.dataset.image_dir, video_id, fname.replace(".png", ".jpg"))
        if not os.path.exists(path):
            return
        image = QImage(path)
        self.image_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))
        self.index = (self.index + 1) % len(self.frames)