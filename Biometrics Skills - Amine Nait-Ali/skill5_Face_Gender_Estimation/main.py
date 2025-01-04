import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget)

from src.AgeEstimatorTab import AgeEstimationTab
from src.GenderEstimationTab import GenderEstimationTab
from src.EmotionRecognitionTab import EmotionRecognitionTab


class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Analysis System")
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(800, 600)

        # Create tab widget
        self.tabs = QTabWidget()
        # self.tabs.addTab(EmotionRecognitionTab(), "Face Emotion Recognition")
        # self.tabs.addTab(AgeEstimationTab(), "Age Estimation")
        self.tabs.addTab(GenderEstimationTab(), "Gender Estimation")
        
        self.setCentralWidget(self.tabs)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

