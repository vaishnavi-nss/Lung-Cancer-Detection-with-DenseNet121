import os
import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
manual_test_folder = "manual_test"

# Load model
def load_model(path="lung_cancer_densenet121.pth"):
    model = models.densenet121(pretrained=False)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_and_show(image_path, model):
    image = Image.open(image_path).convert("RGB")
    
    # Apply transform
    transformed = transform(image)
    if not isinstance(transformed, torch.Tensor):
        raise TypeError("Transform did not return a tensor!")

    input_tensor = transformed.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()

    label = "Cancerous" if predicted_class == 0 else "Non-Cancerous"

    # Show image with title
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"{label} ({confidence:.2f})", fontsize=14, color='green' if label == "Non-Cancerous" else 'red')
    plt.tight_layout()
    plt.show()

    return label, confidence

# Main
if __name__ == "__main__":
    model = load_model()
    print("🧪 Testing Manual Images:\n")

    for file in os.listdir(manual_test_folder):
        if file.endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(manual_test_folder, file)
            print(f"🔍 Predicting: {file}")
            label, conf = predict_and_show(path, model)
            print(f"✅ Prediction: {label} | Confidence: {conf:.2f}\n")
