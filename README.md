# 🧠 MNIST Handwritten Digit Classification using CNN

A Convolutional Neural Network (CNN) project that recognizes handwritten digits from **0 to 9** using the **MNIST dataset**.

The model is trained using TensorFlow/Keras and can predict a digit from a handwritten image provided by the user.


---

## 📌 Project Overview

Handwritten digit recognition is a common image classification problem in Machine Learning and Deep Learning.

In this project, a CNN is trained on the MNIST dataset containing thousands of handwritten digit images. The trained model learns different patterns, edges, curves, and shapes from the images and uses them to classify new handwritten digits.

The final model can take a handwritten digit image as input and predict which digit it represents.

### Example

```text
Handwritten Image
       ↓
Image Preprocessing
       ↓
CNN Model
       ↓
Prediction
       ↓
Digit: 7
```

---

## 📊 Dataset

This project uses the **MNIST handwritten digit dataset**.

The dataset contains:

* **60,000** training images
* **10,000** testing images
* Images are **28 × 28 pixels**
* Images are grayscale
* There are **10 classes**, from 0 to 9

Example classes:

```text
0  1  2  3  4  5
6  7  8  9
```

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* PIL
* Jupyter Notebook
* CNN

---

## 🧠 CNN Architecture

The CNN used in this project contains the following layers:

```text
Input Image
28 × 28 × 1
     ↓
Convolution Layer
32 Filters, 3 × 3
     ↓
ReLU Activation
     ↓
Max Pooling
2 × 2
     ↓
Flatten
     ↓
Dense Layer
128 Neurons
     ↓
Output Layer
10 Neurons
     ↓
Digit Prediction
0 - 9
```

### Layers Used

#### 1. Convolution Layer

The convolution layer extracts important features from the image, such as edges, lines, curves, and shapes.

#### 2. Max Pooling Layer

The max pooling layer reduces the size of the feature maps while keeping important information.

#### 3. Flatten Layer

The extracted features are converted into a one-dimensional array.

#### 4. Dense Layer

The dense layer learns relationships between the extracted features.

#### 5. Output Layer

The final layer contains 10 neurons representing digits from **0 to 9**.

Softmax activation is used to obtain the probability of each digit.

---

## 🔄 Data Preprocessing

The MNIST images are originally represented as:

```text
28 × 28
```

For CNN, the images are reshaped to:

```text
28 × 28 × 1
```

The `1` represents the grayscale channel.

The pixel values are originally between:

```text
0 - 255
```

They are normalized to:

```text
0 - 1
```

using:

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

---

## 🏋️ Model Training

The model is compiled using:

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

The model is then trained using the MNIST training data.

```python
model.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    epochs=8
)
```

---

## 📈 Model Performance

The CNN achieved approximately:

* **Training Accuracy:** 99.76%
* **Best Validation Accuracy:** 98.61%

These results show that the CNN is able to recognize handwritten digits with high accuracy.

---

## 🔮 Prediction

After training, the model can predict a handwritten digit using:

```python
prediction = model.predict(image)
predicted_digit = np.argmax(prediction)

print("Predicted digit:", predicted_digit)
```

For example:

```text
Predicted digit: 7
```

---

## 🖼️ User Image Prediction

The project can also accept a handwritten digit image from the user.

The uploaded image is processed before prediction:

```text
User Image
    ↓
Convert to Grayscale
    ↓
Resize to 28 × 28
    ↓
Normalize Pixel Values
    ↓
Reshape for CNN
    ↓
Model Prediction
    ↓
Predicted Digit
```

This allows the trained CNN to recognize a handwritten digit supplied by the user.

---

## 📁 Project Structure

```text
mnist-cnn-digit-classification/
│
├── notebook.ipynb
├── mnist_cnn.keras
├── app.py
├── requirements.txt
└── README.md
```

> File names can be changed according to the actual files in the project.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/mnist-cnn-digit-classification.git
```

Move into the project directory:

```bash
cd mnist-cnn-digit-classification
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

Example `requirements.txt`:

```text
tensorflow
numpy
matplotlib
pillow
flask
gunicorn
```

---

## 🚀 Future Improvements

Some possible improvements for this project are:

* Create a web interface for digit recognition
* Allow users to draw digits directly on the screen
* Improve preprocessing for real-world handwritten images
* Deploy the model online
* Add prediction confidence
* Improve the CNN architecture
* Add support for larger and more varied handwritten images

---

## 🎯 Applications

Handwritten digit recognition can be used in:

* Digit recognition systems
* Form processing
* Postal code recognition
* Bank cheque processing
* Automatic document processing
* Educational applications
* Optical Character Recognition (OCR)

---

## 📚 Conclusion

This project demonstrates how a **Convolutional Neural Network** can be used for handwritten digit classification.

The CNN learns features from the MNIST dataset and classifies handwritten images into one of ten categories, from **0 to 9**.

The project also demonstrates the complete machine learning workflow:

```text
Dataset
   ↓
Preprocessing
   ↓
CNN Model
   ↓
Training
   ↓
Evaluation
   ↓
Prediction
   ↓
Deployment
```

---

## 👨‍💻 Author

**A Karthik Kumar**

### Project

**MNIST Handwritten Digit Classification using CNN**

