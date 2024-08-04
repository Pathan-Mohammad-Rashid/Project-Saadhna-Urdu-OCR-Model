import torch
from torchvision import transforms
from PIL import Image
from model import UTRNet  # Replace with actual UTRNet model import

# Load UTRNet model
model = UTRNet()
model.load_state_dict(torch.load('models/best_norm_ED.pth'))
model.eval()

def recognize_urdu_text(region_image):
    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize image to fit model input
        transforms.ToTensor(),  # Convert image to tensor
    ])
    input_tensor = preprocess(region_image).unsqueeze(0)  # Add batch dimension

    # Predict using UTRNet
    with torch.no_grad():
        output = model(input_tensor)

    # Decode output to get text
    text = decode_output(output)  # Use a specific function to decode the model's output
    return text
