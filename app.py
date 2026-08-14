from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import tensorflow as tf

app = Flask(__name__)


# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(
    model_path="mnist_cnn.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get image sent from the website
        data = request.get_json()

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

        # Convert to NumPy array
        image_array = np.array(image)

        # ------------------------------------------------
        # Find the handwritten digit
        # ------------------------------------------------

        # Find pixels where the digit exists
        # Background = black (0)
        # Digit = white (> 0)

        rows, cols = np.where(image_array > 20)

        # If nothing was drawn
        if len(rows) == 0:

            return jsonify({
                "digit": 0,
                "confidence": 0
            })


        # Find bounding box of the digit

        top = rows.min()
        bottom = rows.max()

        left = cols.min()
        right = cols.max()


        # Crop the digit

        cropped = image_array[
            top:bottom + 1,
            left:right + 1
        ]


        # Convert cropped image back to PIL
        digit_image = Image.fromarray(cropped)


        # ------------------------------------------------
        # Resize digit while keeping it centered
        # ------------------------------------------------

        # MNIST digits normally occupy roughly 20x20 pixels
        digit_image.thumbnail(
            (20, 20),
            Image.Resampling.LANCZOS
        )


        # Create empty 28x28 black image

        final_image = Image.new(
            "L",
            (28, 28),
            0
        )


        # Calculate position to center digit

        x = (28 - digit_image.width) // 2

        y = (28 - digit_image.height) // 2


        # Put digit in the center

        final_image.paste(
            digit_image,
            (x, y)
        )


        # Convert to NumPy array

        image = np.array(
            final_image
        )


        # Normalize pixel values

        image = image.astype(
            np.float32
        ) / 255.0


        # Reshape for CNN

        image = image.reshape(
            1,
            28,
            28,
            1
        )


        # ------------------------------------------------
        # Run TensorFlow Lite prediction
        # ------------------------------------------------

        interpreter.set_tensor(
            input_details[0]["index"],
            image
        )


        interpreter.invoke()


        # Get prediction

        prediction = interpreter.get_tensor(
            output_details[0]["index"]
        )


        # Find predicted digit

        predicted_digit = int(
            np.argmax(prediction)
        )


        # Find confidence

        confidence = float(
            np.max(prediction)
        ) * 100


        # Send result back to website

        return jsonify({

            "digit": predicted_digit,

            "confidence": round(
                confidence,
                2
            )

        })


    except Exception as e:

        print("Prediction error:", e)

        return jsonify({

            "error": str(e)

        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )
