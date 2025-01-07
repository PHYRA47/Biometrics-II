import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from sklearn.metrics import average_precision_score

# Load pre-trained models
def load_openpose_model():
    # Placeholder for OpenPose model (you can use OpenCV's OpenPose implementation)
    net = cv2.dnn.readNetFromTensorflow("models/graph_opt.pb")
    return net

def load_hrnet_model():
    # Load HRNet model (pretrained on COCO)
    model = torch.hub.load("hrnet", "hrnet", pretrained=True)
    model.eval()
    return model

# Perform pose estimation using OpenPose
def openpose_predict(image, net):
    blob = cv2.dnn.blobFromImage(image, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward()
    return output

# Perform pose estimation using HRNet
def hrnet_predict(image, model):
    image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        output = model(image_tensor)
    return output

# Calculate Mean Average Precision (mAP)
def calculate_mAP(predictions, ground_truth):
    return average_precision_score(ground_truth, predictions)

# Calculate Percentage of Correct Keypoints (PCK)
def calculate_PCK(predictions, ground_truth, threshold=0.5):
    distances = np.linalg.norm(predictions - ground_truth, axis=1)
    return np.mean(distances < threshold)

# Streamlit GUI
def main():
    st.title("Human Body Pose Estimation")
    st.sidebar.header("Options")
    option = st.sidebar.radio("Choose input type:", ("Upload Image", "Use Webcam"))

    # Load models
    openpose_net = load_openpose_model()
    # hrnet_model = load_hrnet_model()

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            image = np.array(image)

            # Perform pose estimation
            openpose_results = openpose_predict(image, openpose_net)
            hrnet_results = hrnet_predict(image, hrnet_model)

            # Display results
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("OpenPose Results:")
            st.image(openpose_results, caption="OpenPose Output", use_column_width=True)
            st.write("HRNet Results:")
            st.image(hrnet_results, caption="HRNet Output", use_column_width=True)

    else:
        st.write("Webcam feed will be displayed here.")
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform pose estimation
            openpose_results = openpose_predict(frame, openpose_net)
            hrnet_results = hrnet_predict(frame, hrnet_model)

            # Display results
            st.image(frame, caption="Webcam Feed", use_column_width=True)
            st.write("OpenPose Results:")
            st.image(openpose_results, caption="OpenPose Output", use_column_width=True)
            st.write("HRNet Results:")
            st.image(hrnet_results, caption="HRNet Output", use_column_width=True)

        cap.release()

if __name__ == "__main__":
    main()