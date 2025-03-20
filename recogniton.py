import numpy as np
import tkinter as tk
import tensorflow as tf  # Import tensorflow for Keras
from tensorflow.keras.models import load_model  # Correct way to import load_model
from PIL import Image, ImageDraw

# Load the trained model
model = load_model('handwritten_digit_recognition_model.h5')

# Set up the Tkinter window
root = tk.Tk()
root.title("Digit Recognition")

# Create a canvas for drawing
canvas_width = 280
canvas_height = 280
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

# Create an image object to draw on
image_pil = Image.new("L", (canvas_width, canvas_height), color=255)  # 'L' mode for grayscale
draw = ImageDraw.Draw(image_pil)

def clear_canvas():
    """Clear the canvas and the drawing."""
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill=255)

def save_and_predict():
    """Save the current drawing, process it, and predict the digit."""
    # Convert the image to a format that the model can process
    img_resized = image_pil.resize((28, 28))  # Resize to 28x28 (model input size)
    img_array = np.array(img_resized)  # Convert to array

    # Normalize and invert the image to match the model's training data
    img_array = 255 - img_array  # Invert colors: white to black, black to white

    # Normalize the image to [0, 1]
    img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
    
    # Ensure correct shape (28,28,1) since model expects grayscale images
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension (grayscale)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict using the trained model
    predictions = model.predict(img_array)
    predicted_digit = np.argmax(predictions[0])

    # Display the predicted result
    result_label.config(text=f"Predicted Digit: {predicted_digit}")

def draw_digit(event):
    """Draw on the canvas based on mouse movements."""
    x, y = event.x, event.y
    radius = 8  # Adjusted thickness for better recognition
    canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill="black", outline="black")
    draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=0)

# Set up the buttons and labels
clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack()

predict_button = tk.Button(root, text="Predict", command=save_and_predict)
predict_button.pack()

result_label = tk.Label(root, text="Predicted Digit: ")
result_label.pack()

# Bind the drawing event
canvas.bind("<B1-Motion>", draw_digit)

# Start the Tkinter event loop
root.mainloop()