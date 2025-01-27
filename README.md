# Pix2Pix GAN

This repository contains code for building the img2img translation model on cityscapes dataset. The notebook contains training pipeline along with prediction function for given image

## Key Features and Functionalities:

- **Dataset Management**: Custom data pipeline to download, preprocess and manage image pairs for pix2pix training, with support for multiple image formats and automatic data augmentation.

- **Data Visualization Tools**: Comprehensive visualization utilities to inspect input-output image pairs, monitor training progress, and analyze model performance through generated samples.

- **U-Net Generator Implementation**: Complete implementation of the U-Net architecture with skip connections, featuring encoder-decoder structure for detailed image translation.

- **PatchGAN Discriminator**: Custom implementation of the PatchGAN discriminator using convolutional layers, optimized for analyzing image patches and maintaining structural consistency.

- **Loss Function Framework**: Implementation of both adversarial and L1 losses, combined to optimize for both perceptual quality and pixel-level accuracy.

- **Training Pipeline**: Robust training infrastructure with customizable hyperparameters, progress tracking, and model checkpointing for reliable training sessions.

- **Inference Module**: Easy-to-use inference pipeline for generating translations from new input images, with support for batch processing and result visualization.

- **Model Export**: Utilities for exporting trained models in various formats, making them ready for deployment in different environments.

- **Documentation and Examples**: Detailed documentation with usage examples, parameter explanations, and best practices for training and inference.

### U-Net Generator:

The U-Net Generator in pix2pix GAN is an encoder-decoder architecture with skip connections, designed for image-to-image translation tasks. The encoder path downsamples the input through convolutional layers, capturing hierarchical features from low-level details to high-level semantics. The decoder path upsamples the features using transposed convolutions, reconstructing the spatial dimensions. The distinctive skip connections directly connect corresponding encoder and decoder layers, preserving fine-grained details that might be lost during compression. This architecture is particularly effective because it combines the deep feature extraction capabilities of a traditional CNN with precise spatial information preservation through skip connections.

