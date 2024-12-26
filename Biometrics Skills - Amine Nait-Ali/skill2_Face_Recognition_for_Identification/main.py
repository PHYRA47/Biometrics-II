import sys
from PyQt5.QtWidgets import QApplication
# from src.app_face_recognition import FaceRecognitionApp
from src.app_DeepFace import FaceRecognitionApp

def main():
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()