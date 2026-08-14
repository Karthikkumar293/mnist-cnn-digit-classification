from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import tensorflow as tf

app = Flask(__name__)


# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="mnist_cnn.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    # Get image from browser
    image_data = data["image"]

    # Remove Base64 header
    image_data = image_data.split(",")[1]

    # Convert Base64 to image
    image_bytes = base64.b64decode(image_data)

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    # Convert to grayscale
    image = image.convert("L")

    # Resize to MNIST size
    image = image.resize((28, 28))

    # Convert to NumPy array
    image = np.array(image)

    # Normalize pixel values
    image = image.astype(np.float32) / 255.0

    # Reshape for CNN
    image = image.reshape(1, 28, 28, 1)

    # Give image to TFLite model
    interpreter.set_tensor(
        input_details[0]["index"],
        image
    )

    # Run prediction
    interpreter.invoke()

    # Get prediction result
    prediction = interpreter.get_tensor(
        output_details[0]["index"]
    )

    # Get predicted digit
    predicted_digit = int(
        np.argmax(prediction)
    )

    # Get confidence
    confidence = float(
        np.max(prediction)
    ) * 100

    return jsonify({
        "digit": predicted_digit,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
