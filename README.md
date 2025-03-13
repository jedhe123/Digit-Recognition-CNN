# Digit Recognition using Convolutional Neural Networks (CNN) #
This project implements a Digit Recognition system using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The model accurately classifies handwritten digits (0-9) from 28x28 grayscale images. Built with TensorFlow/Keras, it achieves high accuracy and supports real-time predictions. The system can be further extended for real-world applications like handwritten document processing.

# Technologies Used #  

Tensorflow

Seaborn

Matplotlib

Numpy 

Pandas


# Requirements #
Ensure the following dependencies are installed before running the project:

```bash
pip install tensorflow numpy matplotlib seaborn
```

# Installation #
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/digit-recognition-cnn.git
   cd digit-recognition-cnn
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

# Model Architecture #
The CNN model follows this architecture:
1. **Input Layer**: 28x28 grayscale image
2. **Conv2D Layer**: 32 filters, 3x3 kernel, ReLU activation
3. **MaxPooling Layer**: 2x2 pool size
4. **Conv2D Layer**: 64 filters, 3x3 kernel, ReLU activation
5. **MaxPooling Layer**: 2x2 pool size
6. **Flatten Layer**
7. **Fully Connected Layer**: 128 neurons, ReLU activation
8. **Output Layer**: 10 neurons (softmax activation for classification)

# Usage #
1. Train the model:
   ```bash
   python train.py
   ```
2. Test the model:
   ```bash
   python test.py
   ```
3. Use the trained model for digit prediction:
   ```bash
   python predict.py --image path/to/image.png
   ```

# Features #
- Uses CNN for high accuracy in digit classification.
  
- Trained on the MNIST dataset.
  
- Implemented using TensorFlow/Keras.
  
- Supports real-time prediction for user-uploaded images.

- Provides visualization of training performance.

# Dataset #
The MNIST dataset consists of:
- 60,000 training images
  
- 10,000 testing images

Each image is a 28x28 grayscale pixel representation of a handwritten digit.

# Model Architecture #
The CNN model follows this architecture:
1. **Input Layer**: 28x28 grayscale image
   
3. **Conv2D Layer**: 32 filters, 3x3 kernel, ReLU activation
   
5. **MaxPooling Layer**: 2x2 pool size
   
7. **Conv2D Layer**: 64 filters, 3x3 kernel, ReLU activation
   
9. **MaxPooling Layer**: 2x2 pool size
    
11. **Flatten Layer**
    
13. **Fully Connected Layer**: 128 neurons, ReLU activation
    
15. **Output Layer**: 10 neurons (softmax activation for classification)


# Results #
The trained CNN model achieves an accuracy of ~99% on the MNIST test dataset.

![{334C5CDC-C89F-486E-9650-4E20A94DDE9D}](https://github.com/user-attachments/assets/40512db0-0967-42ae-804c-31104801a80b)


# Contributions #

Contributions are welcome! Feel free to fork the repository and submit pull requests.

# Contact #

For any issues or suggestions, open an issue or contact -jedhesanjana04@gmail.com



