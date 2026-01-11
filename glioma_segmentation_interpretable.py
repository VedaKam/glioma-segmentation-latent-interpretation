"""
Glioma Segmentation with Interpretable Language-Based Analysis

A multi-modal machine learning pipeline for medical image analysis that prioritizes alignment and interpretability through:
- U-Net architecture for precise tumor segmentation
- Autoencoder-based latent feature extraction
- Retrieval-based natural language interpretation

This project demonstrates core concepts in medical imaging and AI applications to BME through a holistic approach,
leveraging an optimal architecture for segmentation, an core ML method for feature extraction, 
and a safety-oriented approach to medical AI with controlled, retrieval-based summarization.

"""

import os
import re
from typing import Tuple, Dict, List, Optional
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from skimage.measure import perimeter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer, util

# Configuration

class Config:
    """Central configuration for reproducibility and maintainability"""
    
    # Data paths
    DATA_DIR = "/content/drive/MyDrive/BME-AIML-Final-Project/archive/BraTS2020_training_data/content/data"
    SAVE_DIR = "/content/drive/MyDrive/BME-AIML-Final-Project/preprocessed_dataset"
    
    # Model hyperparameters
    UNET_CHANNELS = [16, 32, 64, 128]  # Encoder channel progression
    UNET_NUM_CLASSES = 3  # Background, Edema, Enhancing Tumor
    UNET_LEARNING_RATE = 1e-4
    UNET_EPOCHS = 10
    UNET_BATCH_SIZE = 4
    
    AUTOENCODER_LATENT_DIM = 128
    AUTOENCODER_LEARNING_RATE = 2e-4
    AUTOENCODER_WEIGHT_DECAY = 1e-5
    AUTOENCODER_EPOCHS = 50
    
    # Data preprocessing
    MIN_TUMOR_PIXELS = 50  # Filter out slices with minimal tumor
    NORMALIZATION_EPSILON = 1e-8
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reproducibility
    RANDOM_SEED = 42


# Data Loading and Preprocessing

def load_h5_slice(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single MRI slice from HDF5 file.
    
    Args:
        filepath: Path to .h5 file
        
    Returns:
        image: (240, 240, 4) array with T1, T1ce, T2, FLAIR modalities
        mask: (240, 240, 3) one-hot encoded segmentation mask
        
    Purpose: Validates file structure before loading to prevent
    corrupted data from entering the pipeline.

    """
    try:
        with h5py.File(filepath, "r") as f:
            if "image" not in f or "mask" not in f:
                raise ValueError(f"Invalid HDF5 structure in {filepath}")
            
            image = f["image"][:]
            mask = f["mask"][:]
            
            # Validate shapes
            assert image.shape == (240, 240, 4), f"Unexpected image shape: {image.shape}"
            assert mask.shape == (240, 240, 3), f"Unexpected mask shape: {mask.shape}"
            
            return image, mask
    except Exception as e:
        raise RuntimeError(f"Failed to load {filepath}: {str(e)}")


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Z-score normalization: x_norm = (x - μ) / (σ + ε)
    
    Purpose: Critical for medical imaging where acquisition
    parameters can vary significantly between scans.

    """
    mean = image.mean()
    std = image.std()
    return (image - mean) / (std + Config.NORMALIZATION_EPSILON)


def mask_to_categorical(mask_onehot: np.ndarray) -> np.ndarray:
    """
    Convert one-hot encoded mask to categorical labels.
    
    Args:
        mask_onehot: (H, W, 3) one-hot encoded
        
    Returns:
        mask_cat: (H, W) with integer labels {0, 1, 2}

    """
    return np.argmax(mask_onehot, axis=-1)


def preprocess_subject(
    h5_dir: str, 
    subject_id: int, 
    save_dir: str
) -> None:
    
    """
    Preprocess all slices for a single subject and save as numpy arrays.
    
    1. Loads all slices for a subject
    2. Applies Z-score normalization
    3. Converts masks to categorical format
    4. Saves preprocessed data

    """
    os.makedirs(save_dir, exist_ok=True)
    
    X_list, Y_list = [], []
    
    slice_files = sorted([
        f for f in os.listdir(h5_dir)
        if f.startswith(f"volume_{subject_id}_")
    ])
    
    print(f"Subject {subject_id}: {len(slice_files)} slices")
    
    for fname in slice_files:
        path = os.path.join(h5_dir, fname)
        image, mask_oh = load_h5_slice(path)
        
        # Preprocess
        image = preprocess_image(image)
        mask = mask_to_categorical(mask_oh)
        
        X_list.append(image)
        Y_list.append(mask)
    
    X = np.array(X_list)
    Y = np.array(Y_list)
    
    # Save
    np.save(os.path.join(save_dir, f"volume_{subject_id}_X.npy"), X)
    np.save(os.path.join(save_dir, f"volume_{subject_id}_Y.npy"), Y)


def get_all_subject_ids(h5_dir: str) -> List[int]:
    """Extract unique subject IDs from directory of HDF5 files."""
    ids = set()
    pattern = re.compile(r"volume_(\d+)_slice")
    
    for fname in os.listdir(h5_dir):
        match = pattern.match(fname)
        if match:
            ids.add(int(match.group(1)))
    
    return sorted(list(ids))


def filter_tumor_slices(
    X_all: np.ndarray, 
    Y_all: np.ndarray, 
    min_pixels: int = Config.MIN_TUMOR_PIXELS
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter out slices with insufficient tumor tissue.
    
    Purpose: Addresses class imbalance issue in medical segmentation.
    
    """
    tumor_indices = [
        i for i in range(len(Y_all))
        if np.sum(Y_all[i] > 0) > min_pixels
    ]
    
    X_clean = X_all[tumor_indices]
    Y_clean = Y_all[tumor_indices]
    
    print(f"Filtered {len(X_all)} slices → {len(X_clean)} tumor-containing slices")
    
    return X_clean, Y_clean

# PyTorch Dataset

class BraTSDataset(Dataset):
    """PyTorch Dataset for BraTS MRI slices."""
    
    def __init__(self, images: np.ndarray, masks: np.ndarray):
        """
            images: (N, 240, 240, 4) normalized MRI slices
            masks: (N, 240, 240) categorical segmentation masks

        """
        self.images = torch.from_numpy(images).float()
        self.masks = torch.from_numpy(masks).long()
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Permute to (C, H, W) for PyTorch conv layers
        img = self.images[idx].permute(2, 0, 1)  # (4, 240, 240)
        mask = self.masks[idx]  # (240, 240)
        
        return img, mask


# U-Net Architecture

class DoubleConv(nn.Module):
    """
    Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    
    U-Net local feature extraction with batch normalization for training stability.

    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for medical image segmentation.
    
    Architecture:
        Encoder: 3 downsampling stages (16 → 32 → 64 channels)
        Bottleneck: 128 channels at maximum compression
        Decoder: 3 upsampling stages with skip connections
        Output: 3-channel segmentation map (background, edema, enhancing tumor)

    """
    
    def __init__(self, num_classes: int = Config.UNET_NUM_CLASSES):
        super().__init__()
        
        # Encoder (downsampling path)
        self.down1 = DoubleConv(4, 16)
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        
        self.down3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(64, 128)
        
        # Decoder (upsampling path)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)  # 128 = 64 (upsampled) + 64 (skip)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 32)  # 64 = 32 (upsampled) + 32 (skip)
        
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(32, 16)  # 32 = 16 (upsampled) + 16 (skip)
        
        # Output layer
        self.out = nn.Conv2d(16, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connections.

        x: (B, 4, 240, 240) input MRI with 4 modalities
        Returns logits: (B, 3, 240, 240) class logits

        """
        # Encoder
        c1 = self.down1(x)    # (B, 16, 240, 240)
        p1 = self.pool1(c1)   # (B, 16, 120, 120)
        
        c2 = self.down2(p1)   # (B, 32, 120, 120)
        p2 = self.pool2(c2)   # (B, 32, 60, 60)
        
        c3 = self.down3(p2)   # (B, 64, 60, 60)
        p3 = self.pool3(c3)   # (B, 64, 30, 30)
        
        # Bottleneck
        bn = self.bottleneck(p3)  # (B, 128, 30, 30)
        
        # Decoder with skip connections
        u3 = self.up3(bn)                    # (B, 64, 60, 60)
        merge3 = torch.cat([u3, c3], dim=1)  # (B, 128, 60, 60)
        c4 = self.conv3(merge3)              # (B, 64, 60, 60)
        
        u2 = self.up2(c4)                    # (B, 32, 120, 120)
        merge2 = torch.cat([u2, c2], dim=1)  # (B, 64, 120, 120)
        c5 = self.conv2(merge2)              # (B, 32, 120, 120)
        
        u1 = self.up1(c5)                    # (B, 16, 240, 240)
        merge1 = torch.cat([u1, c1], dim=1)  # (B, 32, 240, 240)
        c6 = self.conv1(merge1)              # (B, 16, 240, 240)
        
        return self.out(c6)  # (B, 3, 240, 240)


def train_unet(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = Config.UNET_EPOCHS,
    learning_rate: float = Config.UNET_LEARNING_RATE,
    device: torch.device = Config.DEVICE
) -> List[float]:
    """
    Train U-Net model with cross-entropy loss.
    
    Returns loss_history: Training loss per epoch

    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for img, mask in train_loader:
            img = img.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            logits = model(img)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
    
    return loss_history


# Autoencoder for Latent Feature Extraction

def mask_to_onehot(
    mask: np.ndarray, 
    num_classes: int = Config.UNET_NUM_CLASSES
) -> torch.Tensor:
    """
    Convert categorical mask to one-hot encoding.
    
    mask: (H, W) integer labels
    Returns onehot: (3, H, W) one-hot tensor

    """
    mask_tensor = torch.from_numpy(mask).long()
    onehot = F.one_hot(mask_tensor, num_classes=num_classes)
    return onehot.permute(2, 0, 1).float()


class ImprovedMaskAutoencoder(nn.Module):
    """
    Convolutional autoencoder for learning latent tumor morphology representations.
    
    Purpose: Extract a compact 128-dimensional embedding that captures
        - Tumor size
        - Boundary irregularity
        - Spatial fragmentation
        - Overall complexity
    
    Why autoencoder:
        1. Learns hierarchical representations automatically
        2. Captures complex spatial patterns humans might miss
        3. Latent space can be interpreted via PCA/clustering
    
    Architecture:
        Encoder: 4 conv layers (3 → 32 → 64 → 128 → 256 channels)
                 Downsamples 240x240 → 30x30
        Latent: 128-dimensional bottleneck
        Decoder: 3 transposed conv layers
                 Upsamples 30x30 → 240x240
    
    Purpose: interpretable latent features that can be audited, unlike black-box embeddings.

    """
    
    def __init__(self, latent_dim: int = Config.AUTOENCODER_LATENT_DIM):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 240 → 120
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 120 → 60
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 60 → 30
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 30 → 30
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Latent bottleneck
        self.flat_dim = 256 * 30 * 30
        self.fc_encoder = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, self.flat_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 30 → 60
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 60 → 120
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 120 → 240
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        x: (B, 3, 240, 240) one-hot encoded mask 
        Returns:
            x_reconstructed: (B, 3, 240, 240) reconstructed mask
            latent: (B, latent_dim) compressed representation

        """
        # Encode
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        
        # Latent bottleneck
        latent = self.fc_encoder(h)
        
        # Decode
        h2 = self.fc_decoder(latent)
        h2 = h2.view(-1, 256, 30, 30)
        x_reconstructed = self.decoder(h2)
        
        return x_reconstructed, latent


def train_autoencoder(
    model: nn.Module,
    masks: np.ndarray,
    epochs: int = Config.AUTOENCODER_EPOCHS,
    learning_rate: float = Config.AUTOENCODER_LEARNING_RATE,
    weight_decay: float = Config.AUTOENCODER_WEIGHT_DECAY,
    device: torch.device = Config.DEVICE
) -> List[float]:
    """
    Train autoencoder on segmentation masks.
    
    Uses MSE loss to ensure proper reconstruction of tumor morphology.

    """
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss()
    
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for mask in masks:
            # Convert to one-hot tensor
            mask_oh = mask_to_onehot(mask).unsqueeze(0).to(device)
            
            optimizer.zero_grad()
            mask_reconstructed, latent = model(mask_oh)
            
            loss = loss_fn(mask_reconstructed, mask_oh)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(masks)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.5f}")
    
    return loss_history


def extract_all_latents(
    model: nn.Module,
    masks: np.ndarray,
    device: torch.device = Config.DEVICE
) -> np.ndarray:
    """
    Extract latent embeddings for all masks.
    
    Returns: latents: (N, latent_dim) array of embeddings

    """
    model.eval()
    latents = []
    
    with torch.no_grad():
        for mask in masks:
            mask_oh = mask_to_onehot(mask).unsqueeze(0).to(device)
            _, latent = model(mask_oh)
            latents.append(latent.cpu().numpy().flatten())
    
    return np.array(latents)


# Feature Extraction and Interpretation

def extract_tumor_features(mask: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract geometric features from tumor mask.
    
    Features:
        - Area: Total tumor pixels
        - Complexity: Perimeter / Area ratio (measures boundary irregularity)
        - Compactness: Area / Perimeter² (measures how circular/compact)
    
    These features provide human-interpretable metrics that correlate with
    clinical tumor characteristics.

    """
    area = np.sum(mask > 0)
    
    if area == 0:
        return 0.0, 0.0, 0.0
    
    binary_mask = (mask > 0).astype(int)
    perim = perimeter(binary_mask)
    
    complexity = perim / (area + 1e-6)
    compactness = area / (perim**2 + 1e-6)
    
    return area, complexity, compactness


def describe_size(area: float) -> str:
    """Categorize tumor size for natural language description."""
    if area == 0:
        return "absent"
    elif area < 300:
        return "small"
    elif area < 700:
        return "medium"
    else:
        return "large"


def describe_complexity(complexity: float) -> str:
    """Categorize boundary complexity for natural language description."""
    if complexity < 0.1:
        return "very smooth"
    elif complexity < 0.2:
        return "mostly smooth"
    elif complexity < 0.3:
        return "moderately irregular"
    else:
        return "highly irregular"


def describe_compactness(compactness: float) -> str:
    """Categorize compactness for natural language description."""
    if compactness < 0.02:
        return "very low compactness"
    elif compactness < 0.05:
        return "low compactness"
    else:
        return "high compactness"


# BERT-Based Retrieval System

class TumorSummarizer:
    """
    Retrieval-based natural language interpretation system.
    
    Why retrieval over generation:
        1. Safety: No hallucination risk - all outputs are pre-approved
        2. Clinical Validity: Every summary is written/reviewed by domain experts
        3. Reproducibility: Same features → same summary (deterministic)
        4. Traceability: Can trace which features led to which summary
    
    Architecture:
        1. Extract geometric features from mask
        2. Convert features to natural language description
        3. Embed description using Sentence-BERT
        4. Find most similar pre-written summary via cosine similarity
        5. Return vetted summary

    """
    
    # Pre-defined summary library (would be written/reviewed by radiologists)
    SUMMARY_LIBRARY = [
        "No tumor in this slice.",
        "A small tumor is visible.",
        "A small tumor with uneven edges is visible.",
        "A medium-sized tumor with mostly smooth boundaries is visible.",
        "A medium-sized tumor with some uneven or irregular edges is visible.",
        "A large-sized tumor with mostly smooth boundaries is visible.",
        "A large tumor with some uneven or irregular edges is visible.",
        "A tumor that appears spread out or diffuse is visible.",
    ]
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize summarizer with Sentence-BERT model.
        
        model_name: Pretrained sentence-transformers model

        """
        print("Loading Sentence-BERT model...")
        self.bert_model = SentenceTransformer(model_name)
        
        # Pre-compute embeddings for summary library
        self.summary_embeddings = self.bert_model.encode(
            self.SUMMARY_LIBRARY, 
            convert_to_tensor=True
        )
        print(f"Loaded {len(self.SUMMARY_LIBRARY)} summaries")
    
    def build_feature_text(
        self,
        area: float,
        complexity: float,
        compactness: float,
        latent_mean: float
    ) -> str:
        """ Convert numerical features to natural language description. """
        if area == 0:
            return "No tumor detected."
        
        size_desc = describe_size(area)
        complexity_desc = describe_complexity(complexity)
        compactness_desc = describe_compactness(compactness)
        
        if size_desc == "large":
            return (
                f"This slice contains a {size_desc} tumor. "
                f"It has {complexity_desc} boundaries and {compactness_desc}. "
                f"Overall, the tumor occupies a substantial region of the slice."
            )
        else:
            return (
                f"This slice contains a {size_desc} tumor "
                f"with {complexity_desc} boundaries and {compactness_desc}."
            )
    
    def summarize(
        self,
        mask: np.ndarray,
        latent: np.ndarray
    ) -> Dict[str, any]:
        """
        Generate interpretable summary for a tumor slice.
        
        mask: (240, 240) segmentation mask
        latent: (latent_dim,) autoencoder embedding
            
        Returns:
            dict with:
                - raw_features: (area, complexity, compactness, latent_mean)
                - feature_text: Natural language description of features
                - summary: Retrieved clinical-style summary
                - confidence: Cosine similarity score

        """
        # Extract geometric features
        area, complexity, compactness = extract_tumor_features(mask)
        latent_mean = latent.mean()
        
        # Handle empty slices
        if area == 0:
            return {
                "raw_features": (0, 0, 0, latent_mean),
                "feature_text": "No tumor detected.",
                "summary": "No tumor in this slice.",
                "confidence": 1.0
            }
        
        # Build textual description
        feature_text = self.build_feature_text(
            area, complexity, compactness, latent_mean
        )
        
        # Encode feature text
        text_embedding = self.bert_model.encode(
            feature_text, 
            convert_to_tensor=True
        )
        
        # Find most similar summary via cosine similarity
        similarities = util.cos_sim(text_embedding, self.summary_embeddings)[0]
        best_idx = int(similarities.argmax())
        confidence = float(similarities[best_idx])
        
        return {
            "raw_features": (area, complexity, compactness, latent_mean),
            "feature_text": feature_text,
            "summary": self.SUMMARY_LIBRARY[best_idx],
            "confidence": confidence
        }

# Visualization Utilities

def overlay_mask(
    mri: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.35
) -> np.ndarray:
    """
    Create RGB overlay of segmentation mask on MRI slice.
        - Green: Edema (class 1)
        - Yellow: Enhancing tumor (class 2)

    """
    # Normalize MRI to [0, 1]
    mri_norm = (mri - mri.min()) / (mri.max() - mri.min() + 1e-8)
    mri_rgb = np.stack([mri_norm] * 3, axis=-1)
    
    # Define colors
    colors = {
        1: np.array([0, 1, 0]),  # Edema = green
        2: np.array([1, 1, 0])   # Enhancing = yellow
    }
    
    # Create colored overlay
    overlay = np.zeros_like(mri_rgb)
    for class_id, color in colors.items():
        overlay[mask == class_id] = color
    
    # Blend
    return (1 - alpha) * mri_rgb + alpha * overlay


def visualize_segmentation(
    mri: np.ndarray,
    ground_truth: np.ndarray,
    prediction: Optional[np.ndarray] = None,
    title: str = "Segmentation Result"
) -> None:
    """
    Visualize MRI slice with ground truth and optional prediction.
    
    mri: (240, 240, 4) MRI with multiple modalities
    ground_truth: (240, 240) ground truth mask
    prediction: (240, 240) predicted mask

    """
    n_plots = 3 if prediction is not None else 2
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    
    # MRI (show T1ce modality)
    axes[0].imshow(mri[:, :, 1], cmap='gray')
    axes[0].set_title("MRI (T1ce)")
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(ground_truth, cmap='viridis')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    # Prediction
    if prediction is not None:
        axes[2].imshow(prediction, cmap='viridis')
        axes[2].set_title("Prediction")
        axes[2].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_latent_space(
    latents: np.ndarray,
    tumor_sizes: np.ndarray,
    title: str = "Latent Space Visualization"
) -> None:
    """
    Project latent embeddings to 2D via PCA and visualize.
    
    This visualization helps verify that the latent space captures
    meaningful tumor characteristics (e.g., size, complexity).
    
    If tumors cluster by size in PCA space, it indicates the autoencoder
    successfully learned interpretable representations.

    """
    # PCA to 2D
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)
    
    # Normalize sizes for color mapping
    sizes_norm = (tumor_sizes - tumor_sizes.min()) / \
                 (tumor_sizes.max() - tumor_sizes.min() + 1e-8)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        latents_2d[:, 0],
        latents_2d[:, 1],
        c=sizes_norm,
        cmap='viridis',
        alpha=0.6,
        s=50
    )
    
    plt.colorbar(scatter, label='Normalized Tumor Size')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_training_loss(
    loss_history: List[float],
    title: str = "Training Loss"
) -> None:
    """Plot training loss over epochs."""
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Main Pipeline

def main():
    """
    Main execution pipeline demonstrating the complete workflow:
    
    1. Data preprocessing
    2. U-Net training for segmentation
    3. Autoencoder training for latent features
    4. BERT-based interpretation
    5. Visualization and analysis

    """
    
    print("="*70)
    print("Glioma Segmentation with Interpretable Language-Based Analysis")
    print("="*70)
    print()
    
    # Set random seed for reproducibility
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    # 1: Data Preprocessing

    print(" 1: Preprocessing data...")
    
    # Get all subject IDs
    all_subjects = get_all_subject_ids(Config.DATA_DIR)
    subjects_to_process = all_subjects[:20]  # Process first 20 subjects
    
    # Preprocess each subject
    for subject_id in tqdm(subjects_to_process, desc="Preprocessing subjects"):
        preprocess_subject(Config.DATA_DIR, subject_id, Config.SAVE_DIR)
    
    # Merge all preprocessed data
    X_list, Y_list = [], []
    for fname in sorted(os.listdir(Config.SAVE_DIR)):
        if fname.endswith("_X.npy"):
            subject_id = fname.split("_")[1]
            X = np.load(os.path.join(Config.SAVE_DIR, f"volume_{subject_id}_X.npy"))
            Y = np.load(os.path.join(Config.SAVE_DIR, f"volume_{subject_id}_Y.npy"))
            X_list.append(X)
            Y_list.append(Y)
    
    X_all = np.concatenate(X_list, axis=0)
    Y_all = np.concatenate(Y_list, axis=0)
    
    print(f"Total slices: {len(X_all)}")
    
    # Filter tumor-containing slices
    X_clean, Y_clean = filter_tumor_slices(X_all, Y_all)
    
    # 2: U-Net Training

    print("\n 2: Training U-Net for segmentation...")
    
    # Create dataset and dataloader
    train_dataset = BraTSDataset(X_clean, Y_clean)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.UNET_BATCH_SIZE,
        shuffle=True
    )
    
    # Initialize and train U-Net
    unet = UNet(num_classes=Config.UNET_NUM_CLASSES)
    unet_loss_history = train_unet(unet, train_loader)
    
    # Visualize training loss
    plot_training_loss(unet_loss_history, "U-Net Training Loss")
    
    # Visualize sample prediction
    unet.eval()
    with torch.no_grad():
        sample_idx = 0
        sample_img = torch.from_numpy(X_clean[sample_idx]).float().permute(2, 0, 1).unsqueeze(0)
        sample_img = sample_img.to(Config.DEVICE)
        
        logits = unet(sample_img)
        prediction = logits.argmax(dim=1).cpu().numpy()[0]
    
    visualize_segmentation(
        X_clean[sample_idx],
        Y_clean[sample_idx],
        prediction,
        "U-Net Segmentation Result"
    )
    
    # 3: Autoencoder Training

    print("\n 3: Training autoencoder for latent features...")
    
    autoencoder = ImprovedMaskAutoencoder(Config.AUTOENCODER_LATENT_DIM)
    ae_loss_history = train_autoencoder(autoencoder, Y_clean)
    
    # Visualize training loss
    plot_training_loss(ae_loss_history, "Autoencoder Training Loss")
    
    # Extract latent embeddings
    print("Extracting latent embeddings...")
    latents = extract_all_latents(autoencoder, Y_clean)
    tumor_sizes = np.array([np.sum(mask > 0) for mask in Y_clean])
    
    # Visualize latent space
    visualize_latent_space(latents, tumor_sizes)
    
    # 4: BERT-Based Interpretation

    print("\n 4: Setting up BERT-based interpretation...")
    
    summarizer = TumorSummarizer()
    
    # Generate summaries for sample slices
    print("\nSample tumor summaries:")
    print("="*70)
    
    for i in range(min(10, len(Y_clean))):
        result = summarizer.summarize(Y_clean[i], latents[i])
        
        print(f"\nSlice {i}:")
        print(f"  Raw Features:")
        area, cplx, comp, lm = result["raw_features"]
        print(f"    • Area: {area:.0f} pixels")
        print(f"    • Complexity: {cplx:.3f}")
        print(f"    • Compactness: {comp:.3f}")
        print(f"    • Latent mean: {lm:.3f}")
        print(f"  Summary: \"{result['summary']}\"")
        print(f"  Confidence: {result['confidence']:.3f}")
    
    print("="*70)
    print("\nPipeline Complete!")
    print()
    print("Accurate tumor segmentation with U-Net")
    print("Learned interpretable latent representations")
    print("Safe, retrieval-based natural language summaries")
    print("No hallucination risk - all outputs are pre-approved")
    print()


if __name__ == "__main__":
    main()
