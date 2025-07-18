from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem, QTreeWidgetItemIterator,
    QVBoxLayout, QHBoxLayout, QComboBox, QMessageBox,
    QProgressBar, QToolButton, QAbstractItemView,
)
from PyQt6.QtGui import QPixmap, QKeyEvent, QGuiApplication
from PyQt6.QtCore import Qt, QThread
from typing import Optional
from utils.state import AppState, natural_sort_key
import os
import re
import json
from collections import defaultdict

from ui.workers.dataset_loader import DatasetLoader
from ui.widgets.clickable_label import ClickableLabel
from ui.dialogs.stats_dialog import StatsDialog
from ui.dialogs.video_player_dialog import VideoPlayerDialog

class AnnotationSelector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Annotation Selector")
        self.state = AppState()
        self.thread = None

        # State for single-mask view
        self.full_panoptic_mask = None
        self.selected_label_item = None
        self.high_coverage_filter_active = False
        self.frame_key_to_item_map = {}
        self.coverage_label = None

        # Resize window to 75% of the screen
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.75), int(screen.height() * 0.75))
        self.setMinimumSize(800, 600)

        self.init_ui()

    def resizeEvent(self, event):
        """ Handle window resize to keep overlay centered. """
        super().resizeEvent(event)
        if hasattr(self, 'loading_label'):
            # Center the label within the central widget's area
            self.loading_label.setGeometry(self.centralWidget().rect())

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

        # New dedicated label for coverage, to be placed below the list
        self.coverage_label = QLabel("Coverage: N/A")
        self.coverage_label.setFixedWidth(170)
        self.coverage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.coverage_label.font()
        font.setItalic(True)
        self.coverage_label.setFont(font)
        # Add a top border for visual separation
        self.coverage_label.setStyleSheet("border-top: 1px solid #c0c0c0; padding-top: 5px; margin-top: 5px;")

        self.select_button = QPushButton("Select ✓")
        self.deselect_button = QPushButton("Deselect ✗")
        self.save_button = QPushButton("Save 💾")
        self.load_button = QPushButton("Load 📂")
        self.clear_button = QPushButton("Clear ✖")
        self.stats_button = QPushButton("Show Stats 📊")
        self.play_video_button = QPushButton("Play Video ▶️")

        self.coverage_filter_button = QPushButton("Coverage > 90%")
        self.coverage_filter_button.setCheckable(True)

        self.select_button.clicked.connect(self.select_current)
        self.deselect_button.clicked.connect(self.deselect_current)
        self.save_button.clicked.connect(self.save_selection) # Keep as is
        self.load_button.clicked.connect(self.on_load_button_clicked) # Use a dedicated handler
        self.clear_button.clicked.connect(self.clear_selections)
        self.stats_button.clicked.connect(self.show_stats)
        self.play_video_button.clicked.connect(self.play_video)
        self.coverage_filter_button.toggled.connect(self.toggle_coverage_filter)

        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.deselect_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.stats_button)
        button_layout.addWidget(self.play_video_button)
        button_layout.addWidget(self.coverage_filter_button)

        self.file_list_widget = QTreeWidget()
        self.file_list_widget.setHeaderHidden(True)
        self.file_list_widget.setMaximumHeight(150)
        self.file_list_widget.itemChanged.connect(self.on_item_changed)
        self.file_list_widget.currentItemChanged.connect(self.on_item_selected)
        self.file_list_widget.setCursor(Qt.CursorShape.PointingHandCursor)
        self.file_list_widget.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.file_list_widget.setStyleSheet("""
            QTreeWidget::item:hover {
                background-color: #A0522D; /* Sienna - a darker brown for hover */
                color: white; /* Ensure text is readable on dark background */
            }
        """)

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

        # Right-side panel for labels and coverage
        right_panel_layout = QVBoxLayout()
        right_panel_layout.setSpacing(0)
        right_panel_layout.setContentsMargins(0, 0, 0, 0)
        right_panel_layout.addWidget(self.label_panel)
        right_panel_layout.addWidget(self.coverage_label)
        image_row.addLayout(right_panel_layout)


        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.image_id_label)
        main_layout.addLayout(image_row)
        main_layout.addLayout(button_layout)

        main_layout.addWidget(QLabel("Files:"))
        main_layout.addWidget(self.file_list_widget)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Add loading label overlay
        self.loading_label = QLabel("Loading dataset, please wait...", self)
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            font-size: 24px;
            font-weight: bold;
        """)
        self.loading_label.hide()

        self.load_selections(show_success_message=True)
        self.refresh_file_list()
        self.update_display()

    def on_label_clicked(self, item: QListWidgetItem):
        """
        Handles clicks on individual labels in the list.
        """
        # The first item in the list is a header, so we subtract 1 to get the correct index
        segment_index = self.label_panel.row(item) - 1

        # Ignore clicks on non-selectable items like the header, separator, or coverage info.
        # This prevents calculating an invalid index (e.g., -1 for the header).
        if not (item.flags() & Qt.ItemFlag.ItemIsSelectable):
            if self.full_panoptic_mask:
                self.mask_image.setPixmap(QPixmap.fromImage(self.full_panoptic_mask))
                self.mask_image.update_scaled_pixmap()
            self.selected_label_item = None
            return

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
            single_mask_img = self.state.dataset.get_single_segment_visualization(fname, segment_index)

            if single_mask_img and not single_mask_img.isNull():
                self.mask_image.setPixmap(QPixmap.fromImage(single_mask_img))
                self.mask_image.update_scaled_pixmap()

    def on_dataset_changed(self, dataset_name):
        """
        Handles dataset selection change by loading the new dataset in a background thread.
        """
        if dataset_name == self.state.current_dataset_name:
            return

        if self.thread and self.thread.isRunning():
            QMessageBox.warning(self, "Busy", "A dataset is already being loaded. Please wait.")
            # Revert the combobox to the currently active dataset
            self.dataset_selector.blockSignals(True)
            self.dataset_selector.setCurrentText(self.state.current_dataset_name)
            self.dataset_selector.blockSignals(False)
            return

        # 1. Fast part: Update state to point to the new dataset and store the old one for error recovery
        previous_dataset_name = self.state.current_dataset_name
        self.state.set_active_dataset(dataset_name)

        # 2. Show loading indicator and disable UI
        self.loading_label.show()
        self.loading_label.raise_()
        self.centralWidget().setDisabled(True)

        # 3. Slow part: Load data in a background thread
        self.thread = QThread()
        self.worker = DatasetLoader(self.state)
        self.worker.moveToThread(self.thread)

        # Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_loading_finished)
        self.worker.error.connect(
            lambda msg: self.on_loading_error(msg, previous_dataset_name)
        )

        # Clean up the thread and worker once done
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.clear_thread_reference)

        self.thread.start()

    def on_loading_finished(self):
        """Called via signal when the background worker is done."""
        # The order is important: load state, refresh UI list, then update display
        self.load_selections(show_success_message=True)
        self.refresh_file_list()
        self.update_display()

        self.loading_label.hide()
        self.centralWidget().setDisabled(False)

    def on_loading_error(self, error_message, previous_dataset_name):
        """Called via signal if the worker encounters an error."""
        self.loading_label.hide()
        self.centralWidget().setDisabled(False)
        QMessageBox.critical(self, "Dataset Load Error", error_message)

        # Revert the UI to the last known good dataset and clear the view.
        # The user will have to re-select the dataset to trigger a new load.
        self.dataset_selector.blockSignals(True)
        self.dataset_selector.setCurrentText(previous_dataset_name)
        self.dataset_selector.blockSignals(False)
        self.state.set_active_dataset(previous_dataset_name)
        self.refresh_file_list()
        self.update_display()

    def clear_thread_reference(self):
        """Set self.thread to None after it has been deleted."""
        self.thread = None

    def _get_frame_key_from_item(self, item: QTreeWidgetItem) -> str:
        """
        Constructs the full, unique frame_key from a QTreeWidget item.
        For video datasets, this is "video_id/filename".
        For image datasets, this is just "filename".
        """
        if not item:
            return ""

        if self.state.dataset.is_video_dataset:
            parent = item.parent()
            if parent:
                video_id = parent.text(0)
                fname = item.text(0)
                return f"{video_id}/{fname}"
        return item.text(0)

    def _get_next_index_for_advance(self):
        """
        Calculates the index of the next visible item to display after a select/deselect action.
        This respects any active filters and works for both image and video datasets.
        """
        file_list = self.state.dataset.file_list
        if not file_list:
            return 0

        num_files = len(file_list)
        current_idx = self.state.current_index
        current_frame_key = self.state.current_filename()

        # --- Improved Video Dataset Logic ---
        if self.state.dataset.is_video_dataset:
            if current_frame_key and '/' in current_frame_key:
                current_video_id, _ = current_frame_key.split('/', 1)

                # Build a mapping from video ID to its list of (index, frame_key)
                video_to_frames = defaultdict(list)
                for idx, frame_key in enumerate(file_list):
                    video_id, _ = frame_key.split('/', 1)
                    video_to_frames[video_id].append((idx, frame_key))

                # Track whether we've passed the current video
                passed_current = False

                for video_id in sorted(video_to_frames.keys(), key=natural_sort_key):
                    if not passed_current:
                        if video_id == current_video_id:
                            passed_current = True
                        continue

                    # Look for the first visible frame in the next video
                    for idx, frame_key in video_to_frames[video_id]:
                        if self.is_file_visible(frame_key):
                            return idx

        # --- Fallback: next visible frame regardless of video ---
        for i in range(1, num_files + 1):
            next_idx = (current_idx + i) % num_files
            fname = file_list[next_idx]
            if self.is_file_visible(fname):
                return next_idx

        # --- Ultimate fallback: stay where we are ---
        return current_idx


    def select_current(self):
        current_fname = self.state.current_filename()
        if not current_fname:
            return
        self.state.selected_files.add(current_fname)
        self.state.current_index = self._get_next_index_for_advance()
        self.refresh_file_list()
        self.update_display()

    def deselect_current(self):
        current_fname = self.state.current_filename()
        if not current_fname:
            return
        self.state.selected_files.discard(current_fname)
        self.state.current_index = self._get_next_index_for_advance()
        self.refresh_file_list()
        self.update_display()

    def navigate_list(self, direction: int):
        """Navigates to the next or previous visible item in the list."""
        num_files = len(self.state.dataset.file_list)
        if num_files == 0:
            return

        current_idx = self.state.current_index

        # Search for the next valid (visible) index
        for i in range(1, num_files + 1):
            next_idx = (current_idx + (i * direction) + num_files) % num_files
            fname = self.state.dataset.file_list[next_idx]

            if self.is_file_visible(fname):
                self.state.current_index = next_idx
                self.update_display()
                # After updating the display, we must also sync the file list's highlight.
                self.update_file_list_selection()
                return

    def is_file_visible(self, fname_to_check: str) -> bool:
        """Checks if a file should be visible based on active filters."""
        if not self.high_coverage_filter_active:
            return True

        # This check is now very fast as coverage is pre-calculated and cached at load time.
        coverage = self.state.coverage_cache.get(fname_to_check)
        return coverage is not None and coverage > 90

    def keyPressEvent(self, event: QKeyEvent):
        if not self.state.dataset.file_list or not self.state.current_filename():
            QMessageBox.critical(self, "Navigation Error", "No current image selected.")
            return
        key = event.key()
        if key == Qt.Key.Key_Right:
            self.navigate_list(1)
        elif key == Qt.Key.Key_Left:
            self.navigate_list(-1)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            fname = self.state.current_filename()
            # Make Enter key behave like the Select/Deselect buttons for consistency
            if fname and fname in self.state.selected_files:
                self.deselect_current()
            else:
                self.select_current()
        else:
            super().keyPressEvent(event)

    def update_display(self):
        fname = self.state.current_filename()
        self.image_id_label.setText(f"Image: {fname}")
        orig_img = self.state.get_original_image(fname)
        mask_img = self.state.get_mask_image(fname)
        labels = self.state.get_labels(fname)

        if not fname:
            QMessageBox.critical(self, "Display Error", "No file is currently selected.")
            # Clear display and return instead of crashing
            self.original_image.clear()
            self.mask_image.clear()
            self.label_panel.clear()
            return

        if orig_img is None or orig_img.isNull():
            QMessageBox.critical(self, "Display Error", f"Original image not found or is invalid for: {fname}")
            return

        if mask_img is None or mask_img.isNull():
            QMessageBox.critical(self, "Display Error", f"Panoptic mask not found or is invalid for: {fname}")
            return

        # Store the full mask and reset the selected label for the new image
        self.full_panoptic_mask = mask_img
        self.selected_label_item = None

        self.original_image.setPixmap(QPixmap.fromImage(orig_img))
        self.original_image.update_scaled_pixmap()

        self.mask_image.setPixmap(QPixmap.fromImage(mask_img))
        self.mask_image.update_scaled_pixmap()

        self.label_panel.clear()
        if labels:
            # Add a header
            header_item = QListWidgetItem("Labels")
            font = header_item.font()
            font.setBold(True)
            header_item.setFont(font)
            header_item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.label_panel.addItem(header_item)

            self.label_panel.addItems(labels)

        # Update the dedicated coverage label
        coverage = self.state.coverage_cache.get(fname)
        if coverage is not None:
            self.coverage_label.setText(f"Coverage: {coverage:.2f}%")
            self.coverage_label.show()
        else:
            # Hide the label if there's no coverage data to avoid showing "N/A"
            self.coverage_label.hide()

        file_list = self.state.dataset.file_list
        if file_list:
            current_idx = self.state.current_index
            total = len(file_list)
            percent = int((current_idx + 1) / total * 100)
            self.progress_bar.setValue(percent)
            self.progress_bar.setToolTip(f"Image {current_idx + 1} / {total}")
        else:
            self.progress_bar.setValue(0)
            self.progress_bar.setToolTip("No images in dataset")

    def refresh_file_list(self):
        self.file_list_widget.blockSignals(True)
        self.file_list_widget.clear()
        self.frame_key_to_item_map.clear()

        is_video = getattr(self.state.dataset, "is_video_dataset", False)

        if is_video:
            video_groups = defaultdict(list)
            for frame_key in self.state.dataset.file_list:
                video_id, fname = frame_key.split('/', 1)
                video_groups[video_id].append((fname, frame_key))

            for video_id, frame_data in sorted(video_groups.items(), key=lambda item: natural_sort_key(item[0])):
                parent = QTreeWidgetItem(self.file_list_widget, [video_id])
                parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsUserCheckable) # Folders are not checkable
                for fname, frame_key in sorted(frame_data, key=lambda x: natural_sort_key(x[0])):
                    child = QTreeWidgetItem(parent, [fname])
                    child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    child.setCheckState(0, Qt.CheckState.Checked if frame_key in self.state.selected_files else Qt.CheckState.Unchecked)
                    self.frame_key_to_item_map[frame_key] = child
        else:
            # For image datasets, the frame_key is the filename
            for frame_key in self.state.dataset.file_list:
                item = QTreeWidgetItem(self.file_list_widget, [frame_key])
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Checked if frame_key in self.state.selected_files else Qt.CheckState.Unchecked)
                self.frame_key_to_item_map[frame_key] = item

        self.apply_view_filters()

        # Sync the highlighted item in the list with the current state
        self.update_file_list_selection()

        selected = len(self.state.selected_files)
        total = len(self.state.dataset.file_list)
        self.count_label.setText(f"Selected: {selected} / {total}")

        self.file_list_widget.blockSignals(False)

    def toggle_coverage_filter(self, checked: bool):
        if checked:
            # A vibrant orange to indicate the filter is active.
            self.coverage_filter_button.setStyleSheet("""
                background-color: #FFA500;
                color: white;
                font-weight: bold;
                border: 1px solid #D35400;
            """)
        else:
            # Revert to the default stylesheet to match other buttons.
            self.coverage_filter_button.setStyleSheet("")

        self.high_coverage_filter_active = checked
        # This is now instantaneous because coverage data is pre-cached at load time.
        self.refresh_file_list()
        # If the current item is now hidden, find the next visible one
        if self.state.current_filename() and not self.is_file_visible(self.state.current_filename()):
            self.navigate_list(1)

    def apply_view_filters(self):
        """Hides or shows items in the file list based on active filters."""
        iterator = QTreeWidgetItemIterator(self.file_list_widget)

        # First pass: hide/show individual file items
        while iterator.value():
            item = iterator.value()
            if item.childCount() == 0:  # It's a file item
                frame_key = self._get_frame_key_from_item(item)
                item.setHidden(not self.is_file_visible(frame_key))
            iterator += 1

        # Second pass (for video datasets): hide parent if all children are hidden
        if getattr(self.state.dataset, "is_video_dataset", False):
            iterator = QTreeWidgetItemIterator(self.file_list_widget)
            while iterator.value():
                item = iterator.value()
                if item.childCount() > 0:  # It's a video folder item
                    all_children_hidden = True
                    for i in range(item.childCount()):
                        if not item.child(i).isHidden():
                            all_children_hidden = False
                            break
                    item.setHidden(all_children_hidden)
                iterator += 1

    def update_file_list_selection(self):
        current_fname = self.state.current_filename()
        if not current_fname: # Nothing to select
            return

        # Use the map for an instantaneous lookup, which is much faster than iterating.
        item = self.frame_key_to_item_map.get(current_fname)
        if not item:
            return # Current item might be filtered out and thus not in the visible list

        # Expand parent if it exists and is not expanded, to ensure the item is visible
        parent = item.parent()
        if parent and not parent.isExpanded():
            parent.setExpanded(True)

        self.file_list_widget.setCurrentItem(item)
        self.file_list_widget.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)

    def on_item_changed(self, item: QTreeWidgetItem, column: int):
        if item.childCount() > 0:  # Ignore folders
            return
        frame_key = self._get_frame_key_from_item(item)
        if item.checkState(0) == Qt.CheckState.Checked:
            self.state.selected_files.add(frame_key)
        else:
            self.state.selected_files.discard(frame_key)
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

        # The frame_key format works for both video and image datasets.
        # We just save the list of unique frame_keys.
        # We now save a dictionary to include the last viewed file for resuming sessions.
        data_to_save = {
            "selected_files": sorted(list(self.state.selected_files)),
            "last_viewed": self.state.current_filename()
        }

        try:
            with open(path, "w") as f:
                json.dump(data_to_save, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save selections to:\n{path}\n\nError: {e}")
            return

        QMessageBox.information(self, "Saved", f"Saved {len(self.state.selected_files)} selections to:\n{path}")

    def on_load_button_clicked(self):
        """
        Handles the manual click of the Load button, shows a success message,
        and refreshes the UI.
        """
        self.load_selections(show_success_message=True)
        self.refresh_file_list()
        self.update_display()

    def load_selections(self, show_success_message: bool = False):
        path = self.selection_file_path()
        try:
            with open(path, "r") as f:
                loaded_data = json.load(f)

                last_viewed_file = None
                loaded_files_list = []

                if isinstance(loaded_data, dict):
                    # New format: {"selected_files": [...], "last_viewed": "..."}
                    loaded_files_list = loaded_data.get("selected_files", [])
                    last_viewed_file = loaded_data.get("last_viewed")
                    if not isinstance(loaded_files_list, list):
                         raise ValueError("The 'selected_files' key must contain a list.")
                elif isinstance(loaded_data, list):
                    # Old format (backward compatibility): [...]
                    loaded_files_list = loaded_data
                else:
                    raise ValueError("Unsupported selection file format. Expected a list or a dictionary.")

                loaded_files = set(loaded_files_list)

                available_files = set(self.state.dataset.file_list)
                matched_files = loaded_files.intersection(available_files)

                if loaded_files and not matched_files:
                    QMessageBox.warning(self, "Load Warning",
                        f"None of the {len(loaded_files)} selections in {os.path.basename(path)} "
                        f"match the current dataset: {self.dataset_selector.currentText()}"
                    )

                self.state.selected_files = matched_files
                if show_success_message:
                    QMessageBox.information(self, "Loaded", f"Loaded {len(matched_files)} selections from:\n{path}")

                # Resume from last viewed file if it exists in the current dataset
                if last_viewed_file and last_viewed_file in available_files:
                    try:
                        self.state.current_index = self.state.dataset.file_list.index(last_viewed_file)
                    except ValueError:
                        # This should not happen due to the 'in' check, but for safety.
                        pass # Keep index at 0

        except FileNotFoundError:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.state.selected_files.clear()
            # No message needed for a new file, it's normal.
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            QMessageBox.critical(self, "Load Error", f"Could not load or parse selection file:\n{path}\n\nError: {e}\n\nStarting with empty selection.")
            self.state.selected_files.clear() # Start fresh on error

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

        # If a folder (video) is clicked, we want to select its first child frame instead.
        # By programmatically setting the current item, we will re-trigger this same
        # signal, but with the actual frame item. The function will then proceed
        # with the frame, ensuring the selection and display are always in sync.
        if current.childCount() > 0:
            child = current.child(0)
            if child:
                self.file_list_widget.setCurrentItem(child)
            return # Stop processing for the folder click; the re-triggered signal will handle it.

        frame_key = self._get_frame_key_from_item(current)
        if frame_key and frame_key in self.state.dataset.file_list and self.state.current_filename() != frame_key:
            self.state.current_index = self.state.dataset.file_list.index(frame_key)
            self.update_display()

    def play_video(self):
        frame_key = self.state.current_filename()
        if not self.state.dataset.is_video_dataset:
            QMessageBox.warning(self, "Not a video dataset", "This feature is only available for video datasets.")
            return
        try:
            video_id, _ = frame_key.split('/', 1)
            player = VideoPlayerDialog(self.state.dataset, video_id)
            player.exec()
        except (ValueError, IndexError):
            QMessageBox.warning(self, "Invalid Frame", f"Could not determine video ID from frame: {frame_key}")


    def show_help(self):
        help_text = """
        <b>How to Use Annotation Selector</b><br><br>
        <b>Navigation:</b><br>
        - Use the <b>→ / ←</b> arrow keys to move between images.<br>
        - Click on a filename in the list below to jump directly to it.<br><br>

        <b>Selection:</b><br>
        - Press <b>Enter</b> or click the <b>Select ✓</b> / <b>Deselect ✗</b> buttons.<br>
        - Use the checkboxes in the file list for manual selection.<br><br>

        <b>Images & Masks:</b><br>
        - Click on the main image or mask to open an enlarged view.<br>
        - In the right-hand label panel, click a label to isolate its corresponding mask. Click the same label again to restore the full view.<br><br>

        <b>Video Datasets:</b><br>
        - For video datasets, files are grouped by video ID in the list.<br>
        - The <b>Play Video ▶️</b> button will appear and can be used to play the current clip.<br><br>

        <b>Saving & Loading:</b><br>
        - <b>Save 💾</b> stores your selections to a JSON file.<br>
        - <b>Load 📂</b> restores selections from the file if it exists.<br>
        - <b>Clear ✖</b> resets all selections for the current dataset.<br>
        - Selections are saved to: <code>selected_annotations/selected_{dataset_name}.json</code><br><br>

        <b>Statistics:</b><br>
        - <b>Show Stats 📊</b> opens a dialog comparing statistics between your selected images and the entire dataset.<br>
        """
        QMessageBox.information(self, "How to Use", help_text)