from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from src.BodyPoseEstimationTab import PoseEstimator
from src.ImageTab import PoseEstimationImageTab
from src.VideoTab import PoseEstimationVideoTab
from src.WebcamTab import PoseEstimationWebcamTab

class MainWindow(QWidget):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.estimator = PoseEstimator()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Create tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(PoseEstimationImageTab(self.estimator), "Image")
        self.tabs.addTab(PoseEstimationVideoTab(self.estimator), "Video")
        self.tabs.addTab(PoseEstimationWebcamTab(self.estimator), "Webcam")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        self.setWindowTitle("Pose Estimation Tool")
        self.resize(800, 600)

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())