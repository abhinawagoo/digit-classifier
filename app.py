import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps

# Define the same model architecture
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# Load the trained model
model = DigitClassifier()
model.load_state_dict(torch.load('digit_classifier.pth'))
model.eval()

# Prediction function
def predict_digit(image_dict):
    if image_dict is None:
        return {}
    
    # Extract image from dictionary
    if isinstance(image_dict, dict):
        image = image_dict.get('composite')
        if image is None:
            return {}
    else:
        image = image_dict
    
    # Convert to PIL Image if it's a numpy array
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Convert to grayscale
    image = image.convert('L')
    
    # IMPORTANT: Invert the image (MNIST expects white digits on black background)
    image = ImageOps.invert(image)
    
    # Resize to 28x28
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Preprocess the image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Transform and add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output[0], dim=0)
    
    # Return as dictionary for Gradio
    return {str(i): float(probabilities[i]) for i in range(10)}

# Create Gradio interface with Sketchpad



with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ✨ Handwritten Digit Classifier")
    gr.Markdown("Draw a digit (0-9) below and see the AI predict it!")
    
    with gr.Row():
        with gr.Column():
            sketchpad = gr.Sketchpad(type="pil", image_mode="L", label="Draw here")
            predict_btn = gr.Button("Predict", variant="primary")
        
        with gr.Column():
            output = gr.Label(num_top_classes=3, label="Prediction")
    
    predict_btn.click(fn=predict_digit, inputs=sketchpad, outputs=output)
    
    gr.Markdown("### Tips:")
    gr.Markdown("- Draw the digit in the center")
    gr.Markdown("- Use black on white background")
    gr.Markdown("- Make it reasonably large")

if __name__ == "__main__":
    demo.launch()





# import gradio as gr
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from PIL import Image

# # Define the same model architecture
# class DigitClassifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = torch.relu(x)
#         x = self.conv2(x)
#         x = torch.relu(x)
#         x = torch.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = torch.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         return x

# # Load the trained model
# model = DigitClassifier()
# model.load_state_dict(torch.load('digit_classifier.pth'))
# model.eval()

# # Prediction function
# def predict_digit(image):
#     if image is None:
#         return {}
    
#     # Preprocess the image
#     transform = transforms.Compose([
#         transforms.Resize((28, 28)),
#         transforms.Grayscale(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])
    
#     # Transform and add batch dimension
#     image_tensor = transform(image).unsqueeze(0)
    
#     # Make prediction
#     with torch.no_grad():
#         output = model(image_tensor)
#         probabilities = torch.softmax(output[0], dim=0)
    
#     # Return as dictionary for Gradio
#     return {str(i): float(probabilities[i]) for i in range(10)}

# # Create Gradio interface
# # Create Gradio interface
# demo = gr.Interface(
#     fn=predict_digit,
#     inputs=gr.Image(type="pil", image_mode="L", sources=["canvas"], label="Draw a digit (0-9)"),
#     outputs=gr.Label(num_top_classes=3, label="Prediction"),
#     title="✨ Handwritten Digit Classifier",
#     description="Draw a digit from 0-9 and watch the AI predict it in real-time!",
#     theme=gr.themes.Soft()
# )

# if __name__ == "__main__":
#     demo.launch()