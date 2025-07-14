from datasets.panoptic_dataset import PanopticDataset
import json
import re

def natural_sort_key(s):
    """
    Sort key for natural string sorting. e.g. "item 2" comes before "item 10"
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

class AppState:
    def __init__(self):
        self.datasets = self._load_datasets_from_config()

        if not self.datasets:
            raise ValueError(
                "No valid datasets found in config.json.\n\n"
                "Please create a 'config.json' file from the 'config.json.template' "
                "and add your dataset paths."
            )
        
        self.image_cache = {}
        initial_dataset_name = list(self.datasets.keys())[0]
        # Set initial dataset and load its data (blocking)
        self.change_dataset(initial_dataset_name)

    def _load_datasets_from_config(self):
        datasets = {}
        try:
            with open("config.json") as f:
                config = json.load(f)
            
            for name, params in config.get("datasets", {}).items():
                datasets[name] = PanopticDataset(name=name, **params)

        except FileNotFoundError:
            print("Warning: config.json not found. Please create it from config.json. template and add your dataset paths.")
            return {}
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse config.json: {e}")
            return {}
        return datasets

    def current_filename(self):
        if not self.dataset or not self.dataset.file_list:
            return ""

        return self.dataset.file_list[self.current_index]

    def get_goal_stats(self):
        return self.dataset.get_goal_stats()

    def get_current_stats(self, selected_files):
        return self.dataset.get_current_stats(selected_files)

    def change_dataset(self, dataset_name):
        """
        Sets the new active dataset and loads its data.
        This is a blocking operation, kept for compatibility.
        The new approach is to use set_active_dataset() and load_active_dataset_data() separately.
        """
        self.set_active_dataset(dataset_name)
        self.load_active_dataset_data()

    def set_active_dataset(self, dataset_name):
        """
        Switches the active dataset reference and resets state.
        This is a fast, non-blocking operation.
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found.")
            
        self.current_dataset_name = dataset_name
        self.dataset = self.datasets[dataset_name]
        
        # Reset state for the new dataset
        self.current_index = 0
        self.selected_files = set()
        if hasattr(self, 'image_cache'):
            self.image_cache.clear()

    def load_active_dataset_data(self):
        """
        Loads data for the currently active dataset from disk.
        This is a slow, blocking operation that should be run in a background thread.
        """
        if self.dataset:
            self.dataset.load()
            # Ensure the file list is always in a predictable, natural order
            if hasattr(self.dataset, 'file_list') and self.dataset.file_list:
                # For video datasets, sort by video ID first, then by frame filename
                # to match the order in the UI's tree view.
                if getattr(self.dataset, "is_video_dataset", False):
                    # Create a list of (video_id, filename) tuples for sorting
                    sortable_list = [
                        (self.dataset.video_map.get(fname, ""), fname)
                        for fname in self.dataset.file_list
                    ]
                    # Sort by video_id (natural sort), then by filename (natural sort)
                    sortable_list.sort(key=lambda x: (natural_sort_key(x[0]), natural_sort_key(x[1])))
                    # Recreate the file_list in the new sorted order
                    self.dataset.file_list = [fname for video_id, fname in sortable_list]
                else:
                    # For image datasets, just sort by filename
                    self.dataset.file_list.sort(key=natural_sort_key)

    def _load_and_cache_image(self, fname):
        if not fname:
            return None, None, [], None
        if fname not in self.image_cache:
            self.image_cache[fname] = self.dataset.load_image(fname)
        return self.image_cache[fname]

    def get_original_image(self, fname=None):
        fname = fname or self.current_filename()
        img, _, _, _ = self._load_and_cache_image(fname)
        return img

    def get_mask_image(self, fname=None):
        fname = fname or self.current_filename()
        _, mask, _, _ = self._load_and_cache_image(fname)
        return mask

    def get_labels(self, fname=None):
        fname = fname or self.current_filename()
        _, _, labels, _ = self._load_and_cache_image(fname)
        return labels
