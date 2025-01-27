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


<p align="center">
  <img src="https://github.com/rpattan1607/pix2pix/raw/main/rep_images/unet-diagram.png" alt="U-Net Generator Architecture" width="800"/>
</p>

### PatchGAN Classifier

The PatchGAN Discriminator is a clever architectural choice in pix2pix GAN that classifies whether image patches are real or fake, rather than looking at the entire image. It consists of a series of convolutional layers that progressively downsample the input (256x256x6) through multiple stages. Each layer applies convolution with stride 2, followed by LeakyReLU activation and BatchNormalization. The final output is a 16x16x1 matrix, where each value represents the authenticity score of a corresponding image patch. This patch-based approach is particularly effective at capturing local textures and styles while using fewer parameters than full-image discrimination.

<p align="center">
  <img src="https://github.com/rpattan1607/pix2pix/blob/main/rep_images/patchgan-diagram.png" width="800"/>
</p>

## Requirements
To execute the project, ensure you have the following dependencies installed:

- **requests**: Library for making HTTP requests and handling APIs.
- **os**: Built-in Python module for interacting with the operating system.
- **random**: Built-in Python module for generating random numbers and selections.
- **numpy**: Library for numerical computations and array operations.
- **matplotlib.pyplot**: Library for creating visualizations and plots.
- **PIL** (for `Image`): Python Imaging Library for image processing and manipulation.
- **torch**: PyTorch library for building and training neural networks.
- **torch.nn**: PyTorch module for defining neural network layers and architectures.
- **torch.optim**: PyTorch module for implementing optimization algorithms.
- **torch.utils.data** (for `Dataset`, `DataLoader`, `random_split`): PyTorch utilities for dataset handling and loading.
- **torchvision.transforms**: PyTorch library for image transformations and preprocessing.
- **torchsummary**: Library for summarizing PyTorch models.

## References 
- https://arxiv.org/abs/1611.07004
- https://phillipi.github.io/pix2pix/
- https://arxiv.org/abs/1505.04597



