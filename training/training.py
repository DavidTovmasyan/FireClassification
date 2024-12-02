import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import os


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, save_path: str, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.sigmoid(outputs).squeeze() > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        train_accuracy = accuracy_score(all_labels, all_preds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                val_loss += loss.item()
                preds = torch.sigmoid(outputs).squeeze() > 0.5
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_preds)
        val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Save the entire model
        torch.save(model, "vgg19_fire_classification_entire_model.pth")
        print(f"The model has been saved to: {save_path}")

if __name__ == '__main__':

    # Set the device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up paths for training and validation datasets
    train_dir = './dataset/train'
    val_dir = './dataset/val'
    save_path = './models'

    # Data Preprocessing and Augmentation
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet statistics
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet statistics
    ])

    # Load datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform_train)
    val_data = datasets.ImageFolder(val_dir, transform=transform_val)

    # DataLoader
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Load pre-trained VGG19 model
    model = models.vgg19(pretrained=True)

    # Freeze all layers except the last fully connected layer
    for param in model.parameters():
        param.requires_grad = False

    # Modify the classifier for binary classification
    model.classifier[6] = nn.Linear(4096, 1)
    # 4096 is the output of the second-to-last layer, change to 1 output (binary classification)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss with logits
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, save_path, num_epochs=10)
