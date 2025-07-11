from datasets.vipseg_dataset import VIPSegDataset
import json

class AppState:
    def __init__(self):
        self.datasets = self._load_datasets_from_config()

        if not self.datasets:
            raise ValueError(
                "No valid datasets found in config.json.\n\n"
                "Please create a 'config.json' file from the 'config.json.template' "
                "and add your dataset paths."
            )
        
        self.current_dataset_name = list(self.datasets.keys())[0]
        self.dataset = self.datasets[self.current_dataset_name]
        self.dataset.load()
        self.selected_files = set()
        self.current_index = 0

    def _load_datasets_from_config(self):
        datasets = {}
        try:
            with open("config.json") as f:
                config = json.load(f)
            
            for name, params in config.get("datasets", {}).items():
                datasets[name] = VIPSegDataset(name=name, **params)

        except FileNotFoundError:
            print("Warning: config.json not found. Please create it from config.json.template and add your dataset paths.")
            return {}
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse config.json: {e}")
            return {}
        return datasets

    def current_filename(self):
        return self.dataset.file_list[self.current_index]

    def get_goal_stats(self):
        return self.dataset.get_goal_stats()

    def get_current_stats(self, selected_files):
        return self.dataset.get_current_stats(selected_files)

    def change_dataset(self, name):
        self.current_dataset_name = name
        self.dataset = self.datasets[name]
        self.dataset.load()
        self.selected_files = set()
        self.current_index = 0

    def get_original_image(self, fname=None):
        fname = fname or self.current_filename()
        img, _, _, _ = self.dataset.load_image(fname)
        return img

    def get_mask_image(self, fname=None):
        fname = fname or self.current_filename()
        _, mask, _, _ = self.dataset.load_image(fname)
        return mask

    def get_labels(self, fname=None):
        fname = fname or self.current_filename()
        _, _, labels, _ = self.dataset.load_image(fname)
        return labels
