from flask import Flask, request, jsonify
import torch
from PIL import Image
import numpy as np
import io
import logging
import torch.nn as nn
from segmentation_models_pytorch import Unet

# Initialize Flask app
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Define the UNet model class
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.model = Unet(
            encoder_name='resnet34', 
            encoder_weights='imagenet', 
            in_channels=12,
            classes=1,
        )

    def forward(self, x):
        return self.model(x)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)

# Load model weights (ensure this path is correct and weights are valid)
model_weights_path = r'C:\Users\malak\Downloads\model_weights.pth'
try:
    model_weights = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(model_weights)
except Exception as e:
    app.logger.error(f"Error loading model weights: {str(e)}")
    raise RuntimeError(f"Error loading model weights: {str(e)}")

model.eval()

# Define image preprocessing function
def preprocess_image(image):
    try:
        image = image.convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Convert HWC to CHW
        return image.to(device)
    except Exception as e:
        app.logger.error(f"Error in preprocessing image: {str(e)}")
        raise ValueError(f"Error in preprocessing image: {str(e)}")

# Define inference function
def predict_mask(image_tensor):
    try:
        with torch.no_grad():
            output = model(image_tensor)
        output = (output > 0.5).float()  # Binarize output
        return output.cpu().numpy().squeeze(0)
    except Exception as e:
        app.logger.error(f"Error during model prediction: {str(e)}")
        raise RuntimeError(f"Error during model prediction: {str(e)}")

# Define route for inference
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            app.logger.error("No image uploaded")
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))

        # Preprocess the image
        img_tensor = preprocess_image(img)

        # Predict mask
        mask = predict_mask(img_tensor)

        # Return the mask as a response
        mask_response = mask.tolist()
        return jsonify({"mask": mask_response})

    except Exception as e:
        app.logger.error(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Main entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
