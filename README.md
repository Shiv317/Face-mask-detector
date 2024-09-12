# Face-mask-detector
**The model has achieved an accuracy of 93.5% on the test set and 99.97% on the training set.**

This project aims to build a CNN model that can detect whether a person is wearing a face mask or not. The model is trained using the **Face Mask Dataset** from Kaggle and achieves high accuracy on detecting face masks in real-time images.

## Project Overview

### Steps Involved:

1. **Dataset Download**: The dataset is downloaded from Kaggle using the Kaggle API, which contains images of people with and without face masks.
2. **Data Preprocessing**: 
   - Images are resized to 128x128 and converted into RGB format.
   - The dataset is split into training and testing sets.
   - Labels are assigned as `1` for images with masks and `0` for images without masks.
3. **Model Building**: 
   - A Convolutional Neural Network (CNN) is built using TensorFlow and Keras with several layers of convolution, max pooling, and dense layers.
   - Dropout layers are added to reduce overfitting.
4. **Model Training**: 
   - The model is trained on the preprocessed data using an Adam optimizer and sparse categorical crossentropy loss.
   - The training is run for 100 epochs with a validation split of 10%.
5. **Prediction Function**: 
   - A custom function allows users to input an image and predict whether the person in the image is wearing a mask or not.

## Dataset

- The dataset contains two sets of images:
  - **With Mask**: 3725 images of people wearing masks.
  - **Without Mask**: 3828 images of people without masks.

You can download the dataset from (https://www.kaggle.com/omkargurav/face-mask-dataset).

## Model Architecture

The model is a Convolutional Neural Network (CNN) consisting of the following layers:

1. **Conv2D Layer**: 32 filters of size (3,3), ReLU activation, input shape of (128,128,3).
2. **MaxPooling2D Layer**: Pool size of (2,2).
3. **Conv2D Layer**: 64 filters of size (3,3), ReLU activation.
4. **MaxPooling2D Layer**: Pool size of (2,2).
5. **Flatten Layer**: Flattens the input.
6. **Dense Layer**: 128 units, ReLU activation.
7. **Dropout Layer**: 50% dropout rate to prevent overfitting.
8. **Dense Layer**: 64 units, ReLU activation.
9. **Dropout Layer**: 50% dropout rate.
10. **Dense Layer**: 2 units (output layer), sigmoid activation for binary classification.

### Compilation:
- **Optimizer**: Adam
- **Loss Function**: Sparse categorical crossentropy
- **Metrics**: Accuracy
