from src.FERTab import FERTab
from PyQt5.QtWidgets import QMainWindow, QTabWidget

class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Analysis System")
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(800, 600)  # Set minimum size for the application window

        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.addTab(FERTab(), "Face Emotion Recognition")
        
        self.setCentralWidget(self.tabs)