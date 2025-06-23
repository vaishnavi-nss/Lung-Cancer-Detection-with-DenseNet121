# 🩺 Lung Cancer Detection with DenseNet121

This project uses deep learning (DenseNet121) to detect lung cancer from CT scan images. It is based on the publicly available dataset from kaggle for 2D X Ray Images and 3D images from LUNA16 dataset and is focused on classifying pulmonary nodules as cancerous or non-cancerous using 2D slices extracted from 3D CT scans.

---

## 🔍 Project Overview

- 📚 **Dataset**: [LUNA16](https://luna16.grand-challenge.org/)
                   (https://www.kaggle.com/datasets/shubham2703/lung-cancer-image-dataset)
- 🧠 **Model**: DenseNet121 (pretrained on ImageNet)
- 📊 **Task**: Binary Classification – Cancer / No Cancer
- 💻 **Framework**: PyTorch
- 📁 **Manual Test Set**: 10 curated images for demonstration and testing

---

## 📁 Folder Structure

lung-cancer-detection/
├── manual_test/ # 10 test images (used for inference demo)
├── main.py # DenseNet121-based model training
├── sample.py # Script to run model on manual_test images
├── lung_cancer_densenet121.pth # Trained model weights
├── requirements.txt # To install all dependencies
└── README.md # Project documentation
|__ eval.py # For ROC curve and AUC score

yaml
Copy
Edit

---

## 🚀 How to Run

1. **Clone the repository**:

```bash
git clone https://github.com/vaishnavi-nss/Lung-Cancer-Detection-with-DenseNet121.git
cd Lung-Cancer-Detection-with-DenseNet121
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run inference on manual test images:

bash
Copy
Edit
python sample.py
🧠 Model Details
Model: DenseNet121

Final layer modified for binary classification

Trained using binary cross-entropy loss and Adam optimizer

Evaluation Metrics:

✅ Accuracy: ~98%
🎯 F1 Score: ~0.94
📈 ROC AUC: ~0.99

📊 Example Results
Image	Prediction
1canc1.png	✅ Cancer Detected
2non.png	✅ No Cancer
4canc.png	✅ Cancer Detected!


🧪 Testing
![Screenshot (282)](https://github.com/user-attachments/assets/f52fc89c-826b-4f49-a8ed-cb8cb14f8fb6)
[Screenshot (284)](https://github.com/user-attachments/assets/eb5bc2d6-c594-450e-820e-efc4dab051fc)
![Screenshot (283)](https://github.com/user-attachments/assets/111b3e52-712f-4fb7-9eff-6f887c08bf19)
