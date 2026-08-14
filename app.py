from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__, template_folder=".")


# Load the trained CNN model
model = tf.keras.models.load_model("mnist_cnn.keras")


@app.route("/")
def home():
    return render_template("karthik.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get data sent from JavaScript
    data = request.get_json()

    # Get the image
    image_data = data["image"]

    # Remove the Base64 image header
    image_data = image_data.split(",")[1]

    # Convert Base64 to image bytes
    image_bytes = base64.b64decode(image_data)

    # Open image
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to grayscale
    image = image.convert("L")

    # Resize to MNIST size
    image = image.resize((28, 28))

    # Convert image to NumPy array
    image = np.array(image)

    # Normalize pixel values
    image = image / 255.0

    # Reshape for CNN
    image = image.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(image, verbose=0)

    # Get predicted digit
    predicted_digit = int(np.argmax(prediction))

    # Get confidence
    confidence = float(np.max(prediction)) * 100

    return jsonify({
        "digit": predicted_digit,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
