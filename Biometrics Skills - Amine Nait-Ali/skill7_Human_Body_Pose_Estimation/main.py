import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget)

from src.BodyPoseEstimationTab import BodyPoseEstimationTab


class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Analysis System")
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(800, 600)

        # Create tab widget
        self.tabs = QTabWidget()
        # self.tabs.addTab(GenderEstimationTab(), "Gender Estimation")
        self.tabs.addTab(BodyPoseEstimationTab(), "Face Pose Estimation")
        
        self.setCentralWidget(self.tabs)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())