# main.py

from PyQt6.QtWidgets import QApplication, QMessageBox
import sys
from ui.annotation_selector import AnnotationSelector

if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        window = AnnotationSelector()
        window.show()
        sys.exit(app.exec())
    except ValueError as e:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setText("Application Startup Error")
        msg.setInformativeText(str(e))
        msg.setWindowTitle("Error")
        msg.exec()
        sys.exit(1)
