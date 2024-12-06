import torch
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
import pandas as pd
import os
from torchvision.datasets.folder import default_loader


def test_model(model, test_loader, device, output_csv_path):
    """
    Test the trained model on the test dataset and save results to a CSV file.

    Args:
        model (torch.nn.Module): The trained Vision Transformer model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): The device (CPU or GPU) to run the test on.
        output_csv_path (str): Path to save the test results as a CSV file.
    """
    model.eval()  # Set model to evaluation mode
    results = []

    with torch.no_grad():
        for inputs, paths in test_loader:
            inputs = inputs.to(device)
            outputs = model(pixel_values=inputs).logits
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()  # Get probabilities

            # Determine predicted class (fire/not fire based on 0.5 threshold)
            preds = (probs > 0.5).astype(int)

            # Store results: Filename and Predicted Label
            for path, pred in zip(paths, preds):
                filename = os.path.basename(path)  # Extract the file name from the full path
                results.append({"Filename": filename, "Label": pred})

    # Save results to CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

if __name__ == '__main__':
    # Set the device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    test_dir = '/home/davtovmas/PycharmProjects/FireClassification/dataset/test'
    model_path = '/home/davtovmas/PycharmProjects/FireClassification/models/vit_fir_detection'
    output_csv_path = '/home/davtovmas/PycharmProjects/FireClassification/results/results_vit.csv'

    # Data Preprocessing
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet statistics
    ])

    # Load test dataset
    class FlatDatasetWithPaths(torch.utils.data.Dataset):
        def __init__(self, root, transform=None):
            self.root = root
            self.transform = transform
            self.image_paths = [
                os.path.join(root, fname)
                for fname in os.listdir(root)
                if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            path = self.image_paths[idx]
            image = default_loader(path)  # Load image
            if self.transform:
                image = self.transform(image)
            return image, path  # Return image tensor and path


    test_data = FlatDatasetWithPaths(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    # Load the trained model
    model = ViTForImageClassification.from_pretrained(
        model_path,  # Path to the fine-tuned model
        num_labels=1  # Binary classification
    )
    model = model.to(device)

    # Test the model
    test_model(model, test_loader, device, output_csv_path)
