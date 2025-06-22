import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm



# ✅ Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔌 Using device:", device)

# ✅ Image transforms (resize to 224x224 for DenseNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to [0,1]
    transforms.Normalize([0.485, 0.456, 0.406],  # mean
                         [0.229, 0.224, 0.225])  # std
])

# ✅ Load dataset from folder
dataset = datasets.ImageFolder(root="data", transform=transform)
print("✅ Class-to-index mapping:", dataset.class_to_idx)

class_names = dataset.classes
print("🔍 Classes found:", class_names)

# ✅ Split into train & val sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
# ✅ Simulate label noise in training data (flip 10% of labels)
import random

def flip_labels_in_subset(subset, flip_percent=0.05):
    """
    subset: torch.utils.data.Subset returned by random_split
    flip_percent: proportion of labels to flip (e.g. 0.05 for 5%)
    """
    num_to_flip = int(len(subset) * flip_percent)
    indices_to_flip = random.sample(range(len(subset)), num_to_flip)

    for idx in indices_to_flip:
        image, label = subset[idx]
        flipped_label = 1 - label  # Flip 0 ↔ 1
        subset.dataset.samples[subset.indices[idx]] = (subset.dataset.samples[subset.indices[idx]][0], flipped_label)

    print(f"⚠️ Injected label noise: flipped {num_to_flip} labels in training set.")

# Call it
flip_labels_in_subset(train_set, flip_percent=0.05)


train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

# ✅ Load DenseNet121 with pretrained weights
model = models.densenet121(pretrained=True)

# ❌ Remove old classifier, 🔁 add new one (2 classes)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model = model.to(device)

# ✅ Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ✅ Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"🧠 Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100 * correct / total
    print(f"✅ Epoch {epoch+1} | Loss: {train_loss:.4f} | Train Accuracy: {acc:.2f}%")

    # 🔍 Validation accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_acc = 100 * correct / total
    print(f"🔎 Validation Accuracy: {val_acc:.2f}%\n")

# 🧠 Save model
torch.save(model.state_dict(), "lung_cancer_densenet121.pth")
print("✅ Model saved as lung_cancer_densenet121.pth")
