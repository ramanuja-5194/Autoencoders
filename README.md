# Autoencoders on Fashion-MNIST

## Project Overview

This repository contains experiments with autoencoder models trained on the Fashion-MNIST dataset. Two autoencoders are trained with different latent dimensions (16 and 48) to study the trade-off between compression and reconstruction quality. Additionally, latent-space interpolations between sample pairs demonstrate how the model transitions between different clothing items.

## Repository Contents

* `Autoencoders.ipynb`: Jupyter notebook with code for model definition, training, evaluation, visualization of reconstructions, and latent-space interpolations.
* `README.md`: Project documentation (this file).

## Notebook Structure

The `Autoencoders.ipynb` notebook is organized into:

1. **Data Loading & Preprocessing**: Downloading Fashion-MNIST, normalizing images, and creating train/test splits.
2. **Model Definitions**:

   * **Autoencoder (latent dim = 16)**
   * **Autoencoder (latent dim = 48)**
3. **Training & Reconstruction**:

   * Training both models with the same architecture except for latent size.
   * Reporting mean squared error (MSE) for training and test sets.
   * Displaying 8 reconstructed training images and 8 reconstructed test images for each model.
4. **Latent-space Interpolation**:

   * Selecting 4 pairs of training samples.
   * Computing 5 evenly spaced latent vectors between each pair.
   * Decoding and displaying interpolated images.
   * Discussing observed trends in smooth transitions between items.
5. **Analysis**:

   * Comparing reconstruction errors and visual quality across latent dimensions.
   * Interpreting how latent dimension affects compression and representation smoothness.

## Hyperparameter Configuration

Key settings used in the notebook:

* **Latent Dimensions**: 16 and 48
* **Encoder/Decoder Architecture**: Fully connected layers with nonlinear activations
* **Optimizer**: Adam with learning rate 0.001
* **Batch Size**: 128
* **Epochs**: 20
* **Loss Function**: Mean Squared Error (MSE)

## Requirements

* Python 3.7+
* PyTorch
* torchvision
* matplotlib
* numpy
* tqdm

## Results Summary

* **Reconstruction Errors**: MSE reported for both latent sizes on train and test sets, showing lower error for higher latent dimension.
* **Visual Reconstructions**: Models with latent dim 48 produce sharper reconstructions, while dim 16 captures coarse structure.
* **Interpolation Trends**: Smooth morphing between items; higher latent dims yield more detailed transitions.

## License

This project is licensed under the MIT License. Feel free to use and modify!
