## Optimizers - 1.ipynb, 2.ipynb
- Implemented Gradient Descent and its variants (Polyak's Momentum, Nesterov's Accelerated Gradient, Adam Optimizer) from scratch.
- Implemented Backpropogation, RProp and QuickProp from scratch.

## Autoencoder for Image Denoising on MNIST

This involves building and training a convolutional autoencoder to denoise MNIST images. The model is trained using noisy versions of MNIST digits to recover the original clean images.


### Key Steps:
1. **Data Preparation**: 
   - Loaded MNIST dataset.
   - Introduced Gaussian noise to create noisy training, validation, and test datasets.
  
2. **Model**: 
   - Built a convolutional autoencoder using PyTorch. 
   - Encoder compresses noisy images into latent representations.
   - Decoder reconstructs the original clean image from the latent representation.

3. **Training**: 
   - Trained the model with noisy inputs and clean labels using Mean Squared Error (MSE) loss for 10 epochs.
  
4. **Evaluation**:
   - Evaluated the performance on noisy test images and visualized the original, noisy, and denoised images.
   - Computed the final test loss on the denoised outputs.

### Requirements:
- PyTorch
- torchvision
- matplotlib
- numpy

