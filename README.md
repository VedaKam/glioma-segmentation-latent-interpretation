# Glioma Segmentation with Interpretable Language-Based Analysis

## Project Overview

This project implements a multi-modal machine learning pipeline for brain tumor (glioblastoma) analysis that **prioritizes alignment and interpretability**. It demonstrates how thoughtful architectural choices can make AI systems more suitable for high-stakes medical applications.

### 1. **Avoiding Hallucination in Safety-Critical Contexts**

Generative models can produce fluent but factually incorrect text ("hallucinations"). In medical settings, this is unacceptable. Our retrieval-based system:

- Only outputs pre-written, expert-reviewed summaries
- Maps tumor features to vetted clinical descriptions via semantic similarity
- Provides deterministic, reproducible outputs for the same input features
- Enables complete auditability: we can trace exactly which features led to which summary

### 2. **Interpretability Through Multi-Stage Design**

```
MRI → U-Net Segmentation → Geometric Features → Text Encoding → Summary Retrieval
                         ↘ Autoencoder Latents ↗
```

Each stage produces human-interpretable intermediate outputs:
- **U-Net**: Visual segmentation masks clinicians can verify
- **Geometric features**: Quantitative metrics (area, perimeter, compactness)
- **Latent embeddings**: Can be visualized via PCA to show learned organization
- **Text retrieval**: Explicit matching scores show confidence

### 3. **Robustness Through Controlled Outputs**

By constraining the output space to a curated library of summaries, we:
- Prevent the system from making unbounded claims
- Ensure clinical terminology is used correctly
- Enable domain experts to review and update all possible outputs
- Reduce the attack surface for adversarial inputs

## Technical Architecture

### U-Net for Segmentation

**Input:** 240×240 MRI slices with 4 modalities (T1, T1ce, T2, FLAIR)  
**Output:** 3-class pixel-wise segmentation (background, edema, enhancing tumor)  
**Loss:** Cross-entropy  

The U-Net's skip connections preserve spatial details essential for precise tumor boundary delineation.

### Convolutional Autoencoder for Latent Features

**Input:** One-hot encoded segmentation masks  
**Latent dimension:** 128  
**Purpose:** Learn compact representations capturing tumor morphology  

The autoencoder discovers hierarchical features (size, boundary irregularity, fragmentation) without explicit programming. PCA visualization confirms the latent space organizes tumors by clinical characteristics.

### BERT-Based Retrieval System

**Approach:** Sentence-BERT embeddings + cosine similarity  
**Library:** 8 pre-written clinical summaries
**Process:**
1. Extract geometric features (area, complexity, compactness)
2. Convert features to natural language description
3. Embed description with Sentence-BERT
4. Find most similar pre-written summary
5. Return vetted summary + confidence score

## Dataset

**Source:** BraTS 2020 (Brain Tumor Segmentation Challenge)  
**Subjects used:** 20 (for demonstration; full dataset has 369)  
**Slices:** ~2,325 total → 335 tumor-containing after filtering  
**Modalities:** T1, T1 contrast-enhanced, T2, T2-FLAIR  

Data preprocessing:
- Z-score normalization per slice (reduces acquisition variability)
- One-hot to categorical conversion for masks
- Filtering of tumor-free slices (>50 tumor pixels threshold)

## Results

### U-Net Performance
- **Training loss:** Decreased from 0.9872 → 0.0216 over 10 epochs
- **Qualitative assessment:** Model captures overall tumor location and extent
- **Known limitation:** Struggles with thin edema boundaries (expected with limited data)

### Autoencoder Performance
- **Reconstruction loss:** Decreased from 0.01770 → 0.00212 over 50 epochs
- **PCA visualization:** Latent space clusters tumors by size/complexity
- **Interpretation:** Learned representations align with clinical features

### Retrieval System Performance
- **Consistency:** Successfully maps features to appropriate summaries
- **No hallucinations:** Only outputs from pre-defined library
- **Example outputs:**
  - Tumor-free slices → "No tumor in this slice"
  - Large irregular tumors → "A large tumor with some uneven or irregular edges is visible"

## Installation & Usage

### Requirements
```bash
pip install torch torchvision numpy matplotlib scikit-learn scikit-image h5py tqdm sentence-transformers
```

### Running the Pipeline

```python
# Mount data (if using Google Colab)
from google.colab import drive
drive.mount('/content/drive')

# Run main pipeline
python glioma_segmentation_interpretable.py
```

The pipeline will:
1. Preprocess MRI data from HDF5 files
2. Train U-Net for segmentation
3. Train autoencoder for latent features
4. Extract features and generate interpretable summaries
5. Visualize results (segmentations, latent space, training curves)

## Limitations & Future Work

### Current Limitations
1. **2D analysis:** Ignores inter-slice spatial continuity present in 3D volumes
2. **Limited class granularity:** Only 3 classes; clinical practice distinguishes necrotic cores, edema subtypes, etc.
3. **Small dataset:** 20 subjects for demonstration (full BraTS has 369)
4. **Simple summary library:** Only 8 templates; could be expanded with clinical input

### Proposed Improvements
1. **3D U-Net or attention mechanisms:** Exploit volumetric spatial relationships
2. **Focal/Dice loss:** Better handle class imbalance than cross-entropy
3. **Expanded summary library:** Cover rarer morphologies, include quantitative bounds
4. **Clinical validation:** Compare retrieved summaries to radiologist annotations
5. **Joint representation learning:** Explicitly align visual and language embeddings while preserving interpretability
6. **Uncertainty quantification:** Bayesian deep learning or ensemble methods for prediction intervals
7. **Adversarial robustness:** Test against perturbations that might manipulate segmentation/retrieval

## Code Structure

```
glioma_segmentation_interpretable.py
│
├── Configuration (Config class)
│   └── Centralized hyperparameters, paths, settings
│
├── Data Loading & Preprocessing
│   ├── load_h5_slice()
│   ├── preprocess_image()
│   ├── preprocess_subject()
│   └── filter_tumor_slices()
│
├── PyTorch Models
│   ├── UNet (segmentation)
│   │   ├── DoubleConv blocks
│   │   ├── Encoder (downsampling)
│   │   ├── Bottleneck
│   │   └── Decoder (upsampling + skip connections)
│   │
│   └── ImprovedMaskAutoencoder (latent features)
│       ├── Convolutional encoder
│       ├── Latent bottleneck (128-D)
│       └── Convolutional decoder
│
├── Feature Extraction
│   ├── extract_tumor_features() - geometric metrics
│   └── Feature categorization functions
│
├── BERT-Based Retrieval
│   └── TumorSummarizer class
│       ├── Pre-written summary library
│       ├── Sentence-BERT encoding
│       └── Cosine similarity retrieval
│
├── Visualization
│   ├── overlay_mask()
│   ├── visualize_segmentation()
│   ├── visualize_latent_space()
│   └── plot_training_loss()
│
└── Main Pipeline (main())
    ├── Step 1: Data preprocessing
    ├── Step 2: U-Net training
    ├── Step 3: Autoencoder training
    ├── Step 4: BERT-based interpretation
    └── Results visualization
```

## Citation & Acknowledgments

This project was developed as part of Cornell University's BME AI/ML course under Professor John Zimmerman.

**Dataset:** BraTS 2020 Challenge  
**Key Libraries:** PyTorch, Sentence-Transformers, scikit-image

## Contact

**Authors:** Veda Kamaraju (vsk32) 
**Institution:** Cornell University

---