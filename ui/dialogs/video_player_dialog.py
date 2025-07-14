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

        # Get frames belonging to this video using the new frame_key format
        self.frames = []
        video_image_dir = os.path.join(dataset.image_dir, video_id)
        for frame_key in dataset.file_list:
            # For video datasets, frame_key is "video_id/fname.ext"
            if frame_key.startswith(f"{video_id}/"):
                fname = frame_key.split('/', 1)[1]
                base_name, _ = os.path.splitext(fname)
                image_path = os.path.join(video_image_dir, f"{base_name}.jpg")
                if os.path.exists(image_path):
                    self.frames.append(fname)
        self.frames.sort()
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
        base_name, _ = os.path.splitext(fname)
        path = os.path.join(self.dataset.image_dir, self.video_id, f"{base_name}.jpg")

        if os.path.exists(path):
            image = QImage(path)
            self.image_label.setPixmap(QPixmap.fromImage(image).scaled(
                self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            ))

        self.index = (self.index + 1) % len(self.frames)