# 🧠 MNIST Handwritten Digit Classification using CNN

A Convolutional Neural Network (CNN) project that recognizes handwritten digits from **0 to 9** using the **MNIST dataset**.

The model is trained using TensorFlow/Keras and deployed as a web application where users can draw a handwritten digit directly on the screen and get the predicted digit with its confidence score.

🚀 **Live Demo:** https://mnist-cnn-digit-classification.onrender.com

---

## 📌 Project Overview

Handwritten digit recognition is a common image classification problem in Machine Learning and Deep Learning.

In this project, a CNN is trained on the MNIST dataset. The model learns different patterns, edges, curves, and shapes from handwritten digit images and uses them to classify new digits.

The project also includes a Flask web application with an interactive drawing canvas. Users can draw a digit from 0 to 9 and receive the model's prediction.

### Example

    User Draws Digit
           ↓
    Image Preprocessing
           ↓
    CNN Model
           ↓
    Prediction
           ↓
    Digit: 7
    Confidence: 98%

---

## 📊 Dataset

This project uses the **MNIST handwritten digit dataset**.

- 60,000 training images
- 10,000 testing images
- Image size: 28 × 28 pixels
- Grayscale images
- 10 classes: 0 to 9

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- CNN
- NumPy
- Pillow
- Flask
- HTML
- CSS
- JavaScript
- TensorFlow Lite
- Gunicorn
- Render

---

## 🧠 CNN Architecture

The CNN model contains:

    Input Image
    28 × 28 × 1
          ↓
    Conv2D
    32 Filters
    3 × 3 Kernel
    ReLU Activation
          ↓
    MaxPooling2D
    2 × 2
          ↓
    Flatten
          ↓
    Dense
    128 Neurons
    ReLU Activation
          ↓
    Dense
    10 Neurons
    Softmax
          ↓
    Predicted Digit

The final layer contains 10 neurons representing digits 0 to 9.

---

## 🔄 How It Works

    User Draws Digit
           ↓
    HTML Canvas
           ↓
    JavaScript
           ↓
    Flask Backend
           ↓
    Image Preprocessing
           ↓
    TensorFlow Lite CNN Model
           ↓
    Prediction
           ↓
    Digit + Confidence

---

## ✍️ Web Application

The web application allows users to draw handwritten digits directly on the screen.

### Features

- Draw handwritten digits
- Clear the canvas
- Predict the digit
- Display prediction confidence
- Interactive web interface
- CNN-based classification
- TensorFlow Lite deployment
  

---

## 🖼️ Image Preprocessing

The user's drawing is processed before being sent to the CNN model.

    User Drawing
          ↓
    Convert to Grayscale
          ↓
    Find Digit Area
          ↓
    Crop Digit
          ↓
    Resize Digit
          ↓
    Center Digit
          ↓
    Convert to 28 × 28
          ↓
    Normalize Pixel Values
          ↓
    CNN Model

This preprocessing helps make the user's drawing more similar to the MNIST images used during training.

---

## 📈 Model Performance

The CNN achieved approximately:

- Training Accuracy: **99%+**
- Validation/Test Accuracy: **98%+**

The model was tested using the MNIST test dataset before deployment.

---

## 📁 Project Structure

    mnist-cnn-digit-classification/
    │
    ├── app.py
    ├── mnist_cnn.keras
    ├── mnist_cnn.tflite
    ├── MNIST_ handwritting_CNN.ipynb
    ├── README.md
    ├── requirements.txt
    ├── .python-version
    │
    └── templates/
        └── index.html

---

## 📄 File Description

| File | Description |
|------|-------------|
| `app.py` | Flask backend for image processing and prediction |
| `mnist_cnn.keras` | Original trained CNN model |
| `mnist_cnn.tflite` | Lightweight TensorFlow Lite model used for deployment |
| `MNIST_ handwritting_CNN.ipynb` | Jupyter Notebook containing model training and testing |
| `templates/index.html` | Web interface and drawing canvas |
| `requirements.txt` | Required Python libraries |
| `.python-version` | Python version used for deployment |
| `README.md` | Project documentation |

---

## ⚙️ Installation

Clone the repository:

    git clone https://github.com/Karthikkumar293/mnist-cnn-digit-classification.git

Go to the project directory:

    cd mnist-cnn-digit-classification

Install the required libraries:

    pip install -r requirements.txt

---

## ▶️ Run Locally

Run the Flask application:

    python app.py

Then open your browser and visit:

    http://127.0.0.1:5000

Draw a digit and click the **Predict** button.

---

## 🌐 Deployment

The application is deployed using **Render**.

### Live Demo

https://mnist-cnn-digit-classification.onrender.com

The deployed application uses the TensorFlow Lite model to reduce memory usage and make deployment easier on a free server.

> Note: The free Render instance may take some time to respond after a period of inactivity.

---

## 🎯 Future Improvements

- Improve prediction accuracy for user-drawn digits
- Add image upload functionality
- Add prediction history
- Improve the drawing experience
- Add better mobile touch support
- Display probabilities for all 10 digits
- Improve image preprocessing
- Add more handwritten datasets

---

## 📚 Learning Outcomes

Through this project, I learned:

- MNIST dataset and image classification
- Image preprocessing
- CNN architecture
- Convolution and pooling
- Model training and evaluation
- Model prediction
- TensorFlow Lite conversion
- Flask API development
- HTML Canvas and JavaScript
- Connecting frontend with a machine learning model
- Deploying a machine learning application using Render

---

## 👨‍💻 Author

**Karthik Kumar**

GitHub: https://github.com/Karthikkumar293

---

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.

---

**Built with Python, TensorFlow, CNN, Flask and JavaScript.**
