from flask import Flask, request, jsonify
import torch
from torchvision.transforms import transforms
from PIL import Image
from AdUtils import AdClassification


app = Flask(__name__)

model = AdClassification(3, 10, 2)
model.load_state_dict(torch.load(r"E:\PyTorch\Final\AdDetection.pt"))


#model = models.resnet18(pretrained=True)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def transform_image(image_bytes):
    image = Image.open(image_bytes)
    return transform(image)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if an image file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if file:
            # Transform the image so it can be fed into our model
            input_tensor = transform_image(file)
            input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

            # Forward pass, get the model output
            with torch.no_grad():
                output = model(input_batch)

            # Get the predicted class with the highest score
            _, predicted_idx = torch.max(output, 1)
            predicted_class = predicted_idx.item()

            return jsonify({'predicted_class': predicted_class})

    return jsonify({'error': 'Invalid method'}), 405

if __name__ == '__main__':
    app.run(debug=True)
