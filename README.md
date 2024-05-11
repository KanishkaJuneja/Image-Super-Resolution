
# Image Super Resolution using Convolutional Neural Networks

This project demonstrates image super-resolution using Convolutional Neural Networks (CNNs) implemented in TensorFlow and Keras. Super-resolution refers to the process of enhancing the resolution of an image, thereby improving its quality.

## Overview

The repository contains code to implement and train a CNN model for image super-resolution. It utilizes a dataset consisting of pairs of low-resolution and high-resolution images. The model is trained to predict high-resolution images from their low-resolution counterparts.

## Dataset

The dataset used for training consists of pairs of low-resolution and high-resolution images. Both types of images are preprocessed and resized to a uniform size of 256x256 pixels. The dataset is stored in two separate directories: one for low-resolution images and another for high-resolution images.

## Model Architecture

The CNN model architecture is based on a combination of convolutional layers, max-pooling layers, dropout layers, and up-sampling layers. The model takes both low-resolution and high-resolution images as input and predicts the corresponding high-resolution image.

## Training

The model is trained using the Adam optimizer with a mean absolute error loss function. Training is performed for a specified number of epochs, with batch size set accordingly. Additionally, the training process includes validation on a separate portion of the dataset to monitor model performance.

## Evaluation

After training, the model's performance is evaluated using various metrics such as Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM). These metrics provide insights into the quality and fidelity of the predicted high-resolution images compared to the ground truth.

## Results

The trained model is capable of generating high-resolution images from their low-resolution counterparts. Sample images from the test dataset are provided alongside their corresponding low-resolution images and predicted high-resolution images. PSNR and SSIM scores are also reported to quantify the quality of the predictions.

## Usage

To use the code and replicate the experiment:

1. Clone the repository to your local machine.
2. Ensure you have TensorFlow and other required dependencies installed.
3. Prepare your dataset of low-resolution and high-resolution image pairs.
4. Modify the code to load your dataset and adjust any hyperparameters as needed.
5. Train the model using your dataset and evaluate its performance.
6. Experiment with different architectures, loss functions, and training strategies to improve results.
