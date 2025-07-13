from PyQt6.QtWidgets import QDialog, QScrollArea, QWidget, QHBoxLayout, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
from datasets.panoptic_dataset import PanopticDataset
from typing import Set


class StatsDialog(QDialog):
    def __init__(self, dataset: PanopticDataset, selected_files: Set[str]):
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

    def plot_all_histograms(self, dataset: PanopticDataset):
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

    def plot_selected_histograms(self, dataset: PanopticDataset, selected_files: Set[str]):
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