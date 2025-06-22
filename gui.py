import streamlit as st
import os
import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc

# --- Setup
st.set_page_config(page_title="Early Stage Lung Cancer Prediction", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #0077b6;'>🩺 Lung Cancer Detection Interface</h1>
    <h4 style='text-align: center;'>Powered by DenseNet121 | PyTorch | Streamlit</h4>
    <hr>
""", unsafe_allow_html=True)

# --- Model & Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet121(pretrained=False)
model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(torch.load("lung_cancer_densenet121.pth", map_location=device))
model.to(device)
model.eval()

# --- Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_mapping = {0: "Cancerous", 1: "Non-Cancerous"}

# --- Predict Single Image
def predict(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    return pred_class, confidence

# --- Manual Test Set Evaluation
def evaluate_manual_test(folder="manual_test"):
    y_true, y_pred, y_scores = [], [], []
    st.subheader("📁 Test Set Predictions")
    image_cols = st.columns(2)

    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    for i, file in enumerate(files):
        true_label = 0 if "cancer" in file.lower() else 1  # Based on filename assumption
        image_path = os.path.join(folder, file)
        image = Image.open(image_path).convert("RGB")
        pred_class, confidence = predict(image)

        y_true.append(true_label)
        y_pred.append(pred_class)
        y_scores.append(confidence)

        with image_cols[i % 2]:
            st.image(image, caption=f"🧪 {file}", use_column_width=True)
            st.markdown(f"**Prediction:** {class_mapping[pred_class]} ({confidence:.2f})")
            st.markdown(f"**Ground Truth:** {class_mapping[true_label]}")
            st.markdown("---")

    return y_true, y_pred, y_scores

# --- Run Evaluation Button
if st.button("🔍 Run Lung Cancer Classification on Test Set"):
    y_true, y_pred, y_scores = evaluate_manual_test()

    # --- Accuracy
    acc = accuracy_score(y_true, y_pred)
    st.success(f"✅ Overall Accuracy: {acc * 100:.2f}%")

    # --- ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("🩺 ROC Curve - Lung Cancer Classification")
    ax.legend(loc="lower right")
    st.pyplot(fig)
