import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_kaggle import SnoringDataset
from model import RecurrentAutoencoder
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define paths and parameters
dataset_path = 'snore_dataset'  # Replace with your actual dataset path
batch_size = 32
num_epochs = 10
learning_rate = 1e-3

# Create dataset and dataloaders
train_dataset = SnoringDataset(dataset_path, split='train')
val_dataset = SnoringDataset(dataset_path, split='val')
test_dataset = SnoringDataset(dataset_path, split='test')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize autoencoder, loss function, optimizer, and scheduler
autoencoder = RecurrentAutoencoder(input_dim=64).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Training and validation loop
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    # Training
    autoencoder.train()
    train_loss = 0
    for batch in train_loader:
        inputs, targets, _ = batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation
    autoencoder.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets, _ = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = autoencoder(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    # Scheduler step
    scheduler.step(val_loss)

    print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Testing and plotting
autoencoder.eval()
test_loss = 0
original_spectrograms, noisy_spectrograms, reconstructed_spectrograms = [], [], []

with torch.no_grad():
    for batch in test_loader:
        inputs, targets, idx = batch
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = autoencoder(inputs)
        loss = criterion(outputs, targets)
        
        test_loss += loss.item()

        original_spectrograms.append(targets.cpu().numpy())
        noisy_spectrograms.append(inputs.cpu().numpy())
        reconstructed_spectrograms.append(outputs.cpu().numpy())

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4f}')

# Convert lists to numpy arrays for easier indexing
original_spectrograms = np.concatenate(original_spectrograms, axis=0)
noisy_spectrograms = np.concatenate(noisy_spectrograms, axis=0)
reconstructed_spectrograms = np.concatenate(reconstructed_spectrograms, axis=0)

# Debug: Print shapes of the spectrogram arrays
print(f'Original Spectrograms Shape: {original_spectrograms.shape}')
print(f'Noisy Spectrograms Shape: {noisy_spectrograms.shape}')
print(f'Reconstructed Spectrograms Shape: {reconstructed_spectrograms.shape}')

# Ensure the spectrogram arrays have the correct shape
if len(original_spectrograms.shape) == 3:
    original_spectrograms = np.expand_dims(original_spectrograms, axis=1)
if len(noisy_spectrograms.shape) == 3:
    noisy_spectrograms = np.expand_dims(noisy_spectrograms, axis=1)
if len(reconstructed_spectrograms.shape) == 3:
    reconstructed_spectrograms = np.expand_dims(reconstructed_spectrograms, axis=1)

# Debug: Print shapes after expanding dimensions
print(f'Original Spectrograms Shape after expanding: {original_spectrograms.shape}')
print(f'Noisy Spectrograms Shape after expanding: {noisy_spectrograms.shape}')
print(f'Reconstructed Spectrograms Shape after expanding: {reconstructed_spectrograms.shape}')

# Plot a few examples
num_examples_to_plot = 3
for i in range(num_examples_to_plot):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_spectrograms[i, 0], origin='lower', aspect='auto')
    plt.title('Original')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_spectrograms[i, 0], origin='lower', aspect='auto')
    plt.title('Noisy')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed_spectrograms[i, 0], origin='lower', aspect='auto')
    plt.title('Reconstructed')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
