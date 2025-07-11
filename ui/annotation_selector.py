import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem, QTreeWidgetItemIterator,
    QVBoxLayout, QHBoxLayout, QComboBox, QMessageBox, QDialog, QScrollArea,
    QSizePolicy, QProgressBar, QToolButton, QAbstractItemView,
)
from PyQt6.QtGui import QPixmap, QKeyEvent, QMouseEvent, QGuiApplication, QImage
from PyQt6.QtCore import Qt, QSize, QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
from utils.state import AppState
import os 
import re
import json
from collections import defaultdict

def natural_sort_key(s):
    """
    Sort key for natural string sorting. e.g. "item 2" comes before "item 10"
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

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


class StatsDialog(QDialog):
    def __init__(self, dataset, selected_files):
        super().__init__()
        self.setWindowTitle("Dataset Statistics: All vs Selected")
        self.resize(1200, 900)
        self.setMinimumSize(800, 600)

        # Main layout inside scrollable container
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        container = QWidget()
        main_layout = QHBoxLayout(container)

        # Full dataset (left column)
        full_layout = QVBoxLayout()
        self.full_canvas1 = FigureCanvas(plt.Figure())
        self.full_canvas2 = FigureCanvas(plt.Figure())
        full_layout.addWidget(self.full_canvas1)
        full_layout.addWidget(self.full_canvas2)

        # Selected dataset (right column)
        sel_layout = QVBoxLayout()
        self.sel_canvas1 = FigureCanvas(plt.Figure())
        self.sel_canvas2 = FigureCanvas(plt.Figure())
        sel_layout.addWidget(self.sel_canvas1)
        sel_layout.addWidget(self.sel_canvas2)

        main_layout.addLayout(full_layout)
        main_layout.addLayout(sel_layout)

        scroll_area.setWidget(container)

        # Set scroll area as layout
        layout = QVBoxLayout()
        layout.addWidget(scroll_area)
        self.setLayout(layout)

        # Plot histograms
        self.plot_all_histograms(dataset)
        self.plot_selected_histograms(dataset, selected_files)

    def plot_all_histograms(self, dataset):
        mask_counts, label_counts = dataset.get_goal_histograms()

        ax1 = self.full_canvas1.figure.subplots()
        ax2 = self.full_canvas2.figure.subplots()

        bins_mask = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]
        labels_mask = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60",
                       "60-70", "70-80", "80-90", "90-100", "100+"]
        counts_mask = np.histogram(mask_counts, bins=bins_mask)[0]
        ax1.bar(labels_mask, counts_mask, color='#F57C00')
        ax1.set_title("All Images: Masks per Image")
        ax1.set_xlabel("Masks/Image")
        ax1.set_ylabel("# Images")
        ax1.tick_params(axis='x', rotation=45)

        bins_label = [0, 5, 10, 15, 20, 25, float('inf')]
        labels_label = ["0–5", "5–10", "10–15", "15–20", "20–25", "25+"]
        counts_label = np.histogram(label_counts, bins=bins_label)[0]
        ax2.bar(labels_label, counts_label, color='orange')
        ax2.set_title("All Images: Unique Labels")
        ax2.set_xlabel("Unique Labels/Image")
        ax2.set_ylabel("# Images")
        ax2.tick_params(axis='x', rotation=45)

        self.full_canvas1.figure.tight_layout()
        self.full_canvas1.draw()
        self.full_canvas2.figure.tight_layout()
        self.full_canvas2.draw()

    def plot_selected_histograms(self, dataset, selected_files):
        mask_counts, label_counts = dataset.get_selected_histograms(selected_files)

        ax1 = self.sel_canvas1.figure.subplots()
        ax2 = self.sel_canvas2.figure.subplots()

        bins_mask = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]
        labels_mask = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60",
                       "60-70", "70-80", "80-90", "90-100", "100+"]
        counts_mask = np.histogram(mask_counts, bins=bins_mask)[0]
        ax1.bar(labels_mask, counts_mask, color='#F57C00')
        ax1.set_title("Selected Images: Masks per Image")
        ax1.set_xlabel("Masks/Image")
        ax1.set_ylabel("# Images")
        ax1.tick_params(axis='x', rotation=45)

        bins_label = [0, 5, 10, 15, 20, 25, float('inf')]
        labels_label = ["0–5", "5–10", "10–15", "15–20", "20–25", "25+"]
        counts_label = np.histogram(label_counts, bins=bins_label)[0]
        ax2.bar(labels_label, counts_label, color='orange')
        ax2.set_title("Selected Images: Unique Labels")
        ax2.set_xlabel("Unique Labels/Image")
        ax2.set_ylabel("# Images")
        ax2.tick_params(axis='x', rotation=45)

        self.sel_canvas1.figure.tight_layout()
        self.sel_canvas1.draw()
        self.sel_canvas2.figure.tight_layout()
        self.sel_canvas2.draw()


class AnnotationSelector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Annotation Selector")
        self.state = AppState()

        # State for single-mask view
        self.full_panoptic_mask = None
        self.selected_label_item = None

        # Resize window to 75% of the screen
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.75), int(screen.height() * 0.75))
        self.setMinimumSize(800, 600)

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        self.dataset_selector = QComboBox()
        self.dataset_selector.addItems(self.state.datasets.keys())
        self.dataset_selector.currentTextChanged.connect(self.on_dataset_changed)
        self.count_label = QLabel("Selected: 0")

        self.viewed_label = QLabel("Viewed:")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)


        top_layout.addWidget(QLabel("Dataset:"))
        top_layout.addWidget(self.dataset_selector)
        top_layout.addStretch()
        top_layout.addWidget(self.count_label)
        top_layout.addSpacing(20)  # Optional spacing
        top_layout.addWidget(self.viewed_label)
        top_layout.addWidget(self.progress_bar)

        # Floating help button (top-right)
        self.help_button = QToolButton()
        self.help_button.setText("❓")
        self.help_button.setToolTip("How to use this tool")
        self.help_button.setFixedSize(28, 28)
        self.help_button.clicked.connect(self.show_help)
        self.help_button.setStyleSheet("""
            QToolButton {
                border: none;
                font-weight: bold;
                font-size: 16px;
                background-color: #e0e0e0;
                border-radius: 14px;
            }
            QToolButton:hover {
                background-color: #d0d0d0;
            }
        """)
        # Add to a layout that pushes it to the right
        top_layout.addWidget(self.help_button)

        self.image_id_label = QLabel(f"Image: {self.state.current_filename()}")
        font = self.image_id_label.font()
        font.setPointSize(12)
        font.setBold(True)
        self.image_id_label.setFont(font)

        self.original_image = ClickableLabel("Original Image")
        self.mask_image = ClickableLabel("Mask")
        # self.label_panel = QLabel()
        self.label_panel = QListWidget()
        self.label_panel.setFixedWidth(170)
        self.label_panel.setWordWrap(True)
        self.label_panel.itemClicked.connect(self.on_label_clicked)

        self.select_button = QPushButton("Select ✓")
        self.deselect_button = QPushButton("Deselect ✗")
        self.save_button = QPushButton("Save 💾")
        self.load_button = QPushButton("Load 📂")
        self.clear_button = QPushButton("Clear ✖")
        self.stats_button = QPushButton("Show Stats 📊")
        self.play_video_button = QPushButton("Play Video ▶️")

        self.select_button.clicked.connect(self.select_current)
        self.deselect_button.clicked.connect(self.deselect_current)
        self.save_button.clicked.connect(self.save_selection)
        self.load_button.clicked.connect(self.load_selections)
        self.clear_button.clicked.connect(self.clear_selections)
        self.stats_button.clicked.connect(self.show_stats)
        self.play_video_button.clicked.connect(self.play_video)

        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.deselect_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.stats_button)
        button_layout.addWidget(self.play_video_button)

        self.file_list_widget = QTreeWidget()
        self.file_list_widget.setHeaderHidden(True)
        self.file_list_widget.setMaximumHeight(150)
        self.file_list_widget.itemChanged.connect(self.on_item_changed)
        self.file_list_widget.currentItemChanged.connect(self.on_item_selected)
        self.file_list_widget.setCursor(Qt.CursorShape.PointingHandCursor)
    
        image_row = QHBoxLayout()
        image_row.setSpacing(10)
        image_row.setContentsMargins(0, 0, 0, 0)

        # Left image layout
        left_container = QVBoxLayout()
        left_container.setContentsMargins(0, 0, 0, 0)
        left_container.setSpacing(0)
        left_container.addWidget(self.original_image)
        image_row.addLayout(left_container, stretch=1)

        # Right image layout
        right_container = QVBoxLayout()
        right_container.setContentsMargins(0, 0, 0, 0)
        right_container.setSpacing(0)
        right_container.addWidget(self.mask_image)
        image_row.addLayout(right_container, stretch=1)

        # Label panel
        image_row.addWidget(self.label_panel)


        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.image_id_label)
        main_layout.addLayout(image_row)
        main_layout.addLayout(button_layout)

        main_layout.addWidget(QLabel("Files:"))
        main_layout.addWidget(self.file_list_widget)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.refresh_file_list()
        self.update_display()
        self.load_selections()

    def on_label_clicked(self, item: QListWidgetItem):
        """
        Handles clicks on individual labels in the list.
        """
        if self.selected_label_item == item:
            # User clicked the same label again, so deselect it and restore the full mask
            if self.full_panoptic_mask:
                self.mask_image.setPixmap(QPixmap.fromImage(self.full_panoptic_mask))
                self.mask_image.update_scaled_pixmap()
            self.selected_label_item = None
        else:
            # User clicked a new label, show the single mask
            self.selected_label_item = item

            fname = self.state.current_filename()
            # The first item in the list is a header, so we subtract 1 to get the correct index
            segment_index = self.label_panel.row(item) - 1

            single_mask_img = self.state.dataset.get_single_segment_visualization(fname, segment_index)

            if single_mask_img and not single_mask_img.isNull():
                self.mask_image.setPixmap(QPixmap.fromImage(single_mask_img))
                self.mask_image.update_scaled_pixmap()

    def on_dataset_changed(self, dataset_name):
        self.state.change_dataset(dataset_name)
        self.load_selections()  
        self.refresh_file_list()
        self.update_display()
        # Show/hide play button if video_map is defined
        self.play_video_button.setVisible(getattr(self.state.dataset, "is_video_dataset", False))

    def select_current(self):
        self.state.selected_files.add(self.state.current_filename())
        self.refresh_file_list()

    def deselect_current(self):
        self.state.selected_files.discard(self.state.current_filename())
        self.refresh_file_list()

    def save_selection(self):
        QMessageBox.information(self, "Saved", f"Saved {len(self.state.selected_files)} selections.")

    def keyPressEvent(self, event: QKeyEvent):
        if not self.state.dataset.file_list:
            return
        key = event.key()
        if key == Qt.Key.Key_Right:
            self.state.current_index = (self.state.current_index + 1) % len(self.state.dataset.file_list)
        elif key == Qt.Key.Key_Left:
            self.state.current_index = (self.state.current_index - 1) % len(self.state.dataset.file_list)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            fname = self.state.current_filename()
            if fname in self.state.selected_files:
                self.state.selected_files.remove(fname)
            else:
                self.state.selected_files.add(fname)
            self.refresh_file_list()
    
        self.update_display()

    def update_display(self):
        fname = self.state.current_filename()
        self.image_id_label.setText(f"Image: {fname}")
        orig_img = self.state.get_original_image(fname)
        mask_img = self.state.get_mask_image(fname)
        labels = self.state.get_labels(fname)

        # Store the full mask and reset the selected label for the new image
        self.full_panoptic_mask = mask_img
        self.selected_label_item = None

        # Sync highlighted item in the list with current image
        file_list = self.state.dataset.file_list
        if file_list:
            self.file_list_widget.blockSignals(True)
            self.update_file_list_selection()
            self.file_list_widget.blockSignals(False)

        self.original_image.setPixmap(QPixmap.fromImage(orig_img))
        self.original_image.update_scaled_pixmap()

        self.mask_image.setPixmap(QPixmap.fromImage(mask_img))
        self.mask_image.update_scaled_pixmap()

        # self.label_panel.setText("Labels:\n" + "\n".join(labels))
        self.label_panel.clear()
        if labels:
            # The last item is coverage, which we'll display but not make clickable
            coverage_text = labels[-1]
            actual_labels = labels[:-1]

            # Add a header
            header_item = QListWidgetItem("Labels")
            font = header_item.font()
            font.setBold(True)
            header_item.setFont(font)
            header_item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.label_panel.addItem(header_item)

            self.label_panel.addItems(actual_labels)

            # Add a separator
            separator = QListWidgetItem("—" * 20)
            separator.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            separator.setFlags(Qt.ItemFlag.NoItemFlags)
            self.label_panel.addItem(separator)

            # Add coverage info as a non-interactive item
            coverage_item = QListWidgetItem(coverage_text)
            coverage_item.setFlags(Qt.ItemFlag.NoItemFlags) # Make it non-selectable
            self.label_panel.addItem(coverage_item)

        if file_list:
            current_idx = self.state.current_index
            total = len(file_list)
            percent = int((current_idx + 1) / total * 100)
            self.progress_bar.setValue(percent)
            self.progress_bar.setToolTip(f"Image {current_idx + 1} / {total}")

    def refresh_file_list(self):
        self.file_list_widget.blockSignals(True)
        self.file_list_widget.clear()

        is_video = getattr(self.state.dataset, "is_video_dataset", False)

        if is_video:
            video_groups = defaultdict(list)
            for fname in self.state.dataset.file_list:
                video_id = self.state.dataset.video_map.get(fname, "Uncategorized")
                video_groups[video_id].append(fname)

            for video_id, fnames in sorted(video_groups.items(), key=lambda item: natural_sort_key(item[0])):
                parent = QTreeWidgetItem(self.file_list_widget, [video_id])
                parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsUserCheckable) # Folders are not checkable
                for fname in sorted(fnames, key=natural_sort_key):
                    child = QTreeWidgetItem(parent, [fname])
                    child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    child.setCheckState(0, Qt.CheckState.Checked if fname in self.state.selected_files else Qt.CheckState.Unchecked)
        else:
            # Fallback for non-video datasets
            for fname in self.state.dataset.file_list:
                item = QTreeWidgetItem(self.file_list_widget, [fname])
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Checked if fname in self.state.selected_files else Qt.CheckState.Unchecked)

        self.update_file_list_selection()
        selected = len(self.state.selected_files)
        total = len(self.state.dataset.file_list)
        self.count_label.setText(f"Selected: {selected} / {total}")

        self.file_list_widget.blockSignals(False)

    def update_file_list_selection(self):
        current_fname = self.state.current_filename()
        if not current_fname:
            return

        iterator = QTreeWidgetItemIterator(self.file_list_widget)
        while iterator.value():
            item = iterator.value()
            # Find the child item with the matching filename
            if item.childCount() == 0 and item.text(0) == current_fname:
                self.file_list_widget.setCurrentItem(item)
                self.file_list_widget.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)
                # Expand its parent to make it visible
                parent = item.parent()
                if parent and not parent.isExpanded():
                    parent.setExpanded(True)
                break
            iterator += 1

    def on_item_changed(self, item: QTreeWidgetItem, column: int):
        if item.childCount() > 0:  # Ignore folders
            return
        fname = item.text(0)
        if item.checkState(0) == Qt.CheckState.Checked:
            self.state.selected_files.add(fname)
        else:
            self.state.selected_files.discard(fname)
        self.count_label.setText(f"Selected: {len(self.state.selected_files)} / {len(self.state.dataset.file_list)}")

    def show_stats(self):
        if hasattr(self.state.dataset, 'get_goal_histograms'):
            dialog = StatsDialog(self.state.dataset, self.state.selected_files)
            dialog.exec()
        else:
            QMessageBox.information(self, "Not Supported", "This dataset does not support histogram statistics.")

    def selection_file_path(self):
        dataset_name = self.dataset_selector.currentText()
        safe_name = dataset_name.replace(" ", "_").lower()
        folder = "selected_annotations"
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"selected_{safe_name}.json")

    # Saving and Loading Logic
    def save_selection(self):
        path = self.selection_file_path()
        
        confirm = QMessageBox.question(
            self, "Confirm Save",
            f"Are you sure you want to save {len(self.state.selected_files)} selections?\n\n"
            f"This will overwrite the existing file:\n{os.path.basename(path)}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.No:
            return

        is_video = getattr(self.state.dataset, "is_video_dataset", False)
        data_to_save = None

        if is_video:
            # For video datasets, group selected files by their video ID
            video_groups = defaultdict(list)
            for fname in sorted(list(self.state.selected_files)):
                video_id = self.state.dataset.video_map.get(fname)
                if video_id:
                    video_groups[video_id].append(fname)
            data_to_save = video_groups
        else:
            # For image datasets, save a simple list of filenames
            data_to_save = sorted(list(self.state.selected_files))

        try:
            with open(path, "w") as f:
                json.dump(data_to_save, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save selections to:\n{path}\n\nError: {e}")
            return

        QMessageBox.information(self, "Saved", f"Saved {len(self.state.selected_files)} selections to:\n{path}")

    def load_selections(self):
        path = self.selection_file_path()
        try:
            with open(path, "r") as f:
                loaded_data = json.load(f)
                loaded_files = set()
                if isinstance(loaded_data, dict):  # Video dataset format
                    for fnames in loaded_data.values():
                        loaded_files.update(fnames)
                elif isinstance(loaded_data, list):  # Image dataset format
                    loaded_files = set(loaded_data)

                available_files = set(self.state.dataset.file_list)
                self.state.selected_files = loaded_files.intersection(available_files)
                QMessageBox.information(self, "Loaded", f"Loaded {len(self.state.selected_files)} selections from:\n{path}")
        except FileNotFoundError:
            self.state.selected_files.clear() # Silently handle no file on first load
        except (json.JSONDecodeError, TypeError) as e:
            QMessageBox.warning(self, "Load Error", f"Could not parse selection file:\n{path}\n\nError: {e}")
            self.state.selected_files.clear()

        self.refresh_file_list()
        self.play_video_button.setVisible(getattr(self.state.dataset, "is_video_dataset", False))

    def clear_selections(self):
        confirm = QMessageBox.question(
            self, "Confirm Clear",
            "Are you sure you want to clear all selections?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.state.selected_files.clear()
            self.refresh_file_list()
            QMessageBox.information(self, "Cleared", "All selections have been cleared.")

    def on_item_selected(self, current: QTreeWidgetItem, previous: QTreeWidgetItem):
        if not current:
            return

        # If a folder (video) is clicked, select its first child (frame).
        if current.childCount() > 0:
            current = current.child(0)
            if not current:  # Should not happen with current logic, but good practice
                return

        fname = current.text(0)
        if fname in self.state.dataset.file_list and self.state.current_filename() != fname:
            self.state.current_index = self.state.dataset.file_list.index(fname)
            self.update_display()

    def play_video(self):
        fname = self.state.current_filename()
        video_id = self.state.dataset.video_map.get(fname)
        if not video_id:
            QMessageBox.warning(self, "Not a video file", "This file is not part of a video.")
            return
        player = VideoPlayerDialog(self.state.dataset, video_id)
        player.exec()

    def show_help(self):
        help_text = """
            <b>How to Use Annotation Selector</b><br><br>
            <b>Dataset Selection:</b><br>
            - Use the dropdown at the top to switch between datasets.<br><br>

            <b>Navigation:</b><br>
            - Use <b>→ / ←</b> keys to navigate between images.<br>
            - The file list highlights the image currently being viewed.<br><br>

            <b>Selection:</b><br>
            - Press <b>Enter</b> or click <b>Select ✓</b> / <b>Deselect ✗</b> to toggle.<br>
            - Use checkboxes in the list to manually select images.<br><br>

            <b>Saving:</b><br>
            - <b>Save 💾</b> stores selections to disk (per dataset).<br>
            - <b>Load 📂</b> restores selections if available.<br>
            - <b>Clear ✖</b> resets selections.<br><br>

            <b>Stats:</b><br>
            - <b>Show Stats 📊</b> compares selected vs all image stats.<br><br>

            <b>Images:</b><br>
            - Click on images to enlarge them in a separate viewer.<br>
            - Masks can be selected/deselected individually on label click.<br><br>

            Selections are saved to: <code>selected_annotations/selected_{dataset_name}.json</code>
            """
        QMessageBox.information(self, "How to Use", help_text)


class VideoPlayerDialog(QDialog):
    def __init__(self, dataset, video_id):
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
