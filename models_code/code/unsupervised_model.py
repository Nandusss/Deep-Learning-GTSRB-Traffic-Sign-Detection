import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import zipfile
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.multiprocessing import freeze_support

# Set the path directly
DATASET_PATH = r"D:\COLLEGE\FALL_2024\COMP263 DL\FINAL PROJECT\Deep-Learning-GTSRB-Traffic-Sign-Detection-main\models_code\dataset\archive.zip"

class GTSRBDataset(Dataset):
    def __init__(self, zip_path, csv_name, transform=None):
        self.zip_path = zip_path
        self.transform = transform or transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        
        # Read CSV files from zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open(csv_name) as csv_file:
                self.data = pd.read_csv(csv_file)
            with zip_ref.open('Meta.csv') as meta_file:
                self.meta = pd.read_csv(meta_file)
                
        print(f"Loaded {csv_name} with {len(self.data)} images")
        print(f"Number of classes: {len(self.meta)}")
        
        # Print first few rows of data to verify structure
        print("\nFirst few rows of data:")
        print(self.data.head())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                # The Path column should already contain the full path
                img_path = row['Path']
                with zip_ref.open(img_path) as image_file:
                    image = Image.open(image_file).convert('RGB')
        except KeyError as e:
            print(f"Error accessing image at index {idx}")
            print(f"Attempted path: {img_path}")
            print(f"Available paths (first 5):")
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                print("\n".join(list(zip_ref.namelist())[:5]))
            raise e
        
        if self.transform:
            image = self.transform(image)
        
        return image, row['ClassId']

class SimpleAutoEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(SimpleAutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (32, 8, 8)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_latent_features(self, x):
        return self.encoder(x)

def run_training():
    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Set device
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create dataset and data loader
    print("\nLoading dataset...")
    dataset = GTSRBDataset(zip_path=DATASET_PATH, csv_name='Train.csv')
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    # Initialize model
    model = SimpleAutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train autoencoder
    print("\nTraining autoencoder...")
    num_epochs = 10
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for data, _ in progress_bar:
            data = data.to(device)
            
            output = model(data)
            loss = criterion(output, data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        print(f'\nEpoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

    # Extract and save features
    print("\nExtracting features...")
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for data, label in tqdm(train_loader, desc='Extracting features'):
            data = data.to(device)
            latent = model.get_latent_features(data)
            features.append(latent.cpu().numpy())
            labels.append(label.numpy())

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    # Save features
    np.save(os.path.join(results_dir, 'extracted_features.npy'), features)
    np.save(os.path.join(results_dir, 'feature_labels.npy'), labels)

    # Visualize features
    print("\nCreating visualization...")
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab20')
    plt.colorbar(scatter)
    plt.title('PCA visualization of learned features')
    plt.savefig(os.path.join(results_dir, 'feature_visualization.png'))
    plt.close()

    print(f"\nResults saved in: {results_dir}")
    print(f"Extracted features shape: {features.shape}")
    print(f"Number of classes: {len(np.unique(labels))}")

if __name__ == '__main__':
    freeze_support()
    run_training()