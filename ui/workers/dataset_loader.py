from PyQt6.QtCore import QObject, pyqtSignal
from utils.state import AppState

class DatasetLoader(QObject):
    """
    Worker object to load dataset data in a background thread.
    """
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state

    def run(self):
        """The long-running task."""
        try:
            self.state.load_active_dataset_data()
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"An error occurred while loading dataset: {e}")