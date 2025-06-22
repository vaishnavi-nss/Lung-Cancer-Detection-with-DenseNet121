import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report
from torchvision import models

# 🔧 Load trained DenseNet121 model
def load_model(model_path, device):
    model = models.densenet121(pretrained=False)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 🧪 Evaluate model and collect labels & probabilities
def evaluate_model(model, dataloader, device):
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Prob for class 1 (cancerous)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_probs)

# 📊 Plot ROC curve
def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("🔍 ROC Curve - Lung Cancer Detection")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_curve_lung_cancer.png")
    plt.show()

    print(f"\n🧠 AUC Score: {roc_auc:.4f}")

# 📈 Print classification metrics
def print_classification_metrics(y_true, y_scores):
    preds = (y_scores >= 0.5).astype(int)
    print("\n📋 Classification Report:")
    print(classification_report(y_true, preds, target_names=["non-cancerous", "cancerous"]))

# 🚀 Main function
def main():
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔁 Dataset config (use val_loader or test_loader)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root="data", transform=transform)
    val_size = int(0.2 * len(dataset))
    _, val_set = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])
    val_loader = DataLoader(val_set, batch_size=32)

    model = load_model("lung_cancer_densenet121.pth", device)
    y_true, y_scores = evaluate_model(model, val_loader, device)

    plot_roc_curve(y_true, y_scores)
    print_classification_metrics(y_true, y_scores)

if __name__ == "__main__":
    main()
