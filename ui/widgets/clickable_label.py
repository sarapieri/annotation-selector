from PyQt6.QtWidgets import QLabel, QSizePolicy, QDialog, QScrollArea, QVBoxLayout
from PyQt6.QtGui import QPixmap, QMouseEvent
from PyQt6.QtCore import Qt, QSize


class ClickableLabel(QLabel):
    def __init__(self, name=""):
        super().__init__()
        self.name = name
        self._pixmap = None

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def setPixmap(self, pixmap: QPixmap):
        if pixmap and not pixmap.isNull():
            self._pixmap = pixmap
            self.update_scaled_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_scaled_pixmap()


    def update_scaled_pixmap(self):
        if not self._pixmap or self._pixmap.isNull():
            return

        # Cap maximum scaled size to avoid crashes
        widget_size = self.size()
        max_size = QSize(10000, 10000)
        safe_size = widget_size.boundedTo(max_size)

        if safe_size.width() < 10 or safe_size.height() < 10:
            return

        scaled_pixmap = self._pixmap.scaled(
            safe_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        super().setPixmap(scaled_pixmap)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._pixmap:
            self.show_enlarged(self._pixmap)

    def show_enlarged(self, pixmap: QPixmap):
        dialog = QDialog()
        dialog.setWindowTitle(f"Enlarged View: {self.name}")

        label = QLabel()
        label.setPixmap(pixmap.scaled(
            1200, 1000,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

        scroll = QScrollArea()
        scroll.setWidget(label)

        layout = QVBoxLayout()
        layout.addWidget(scroll)
        dialog.setLayout(layout)
        dialog.resize(1200, 1000)
        dialog.exec()