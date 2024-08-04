# from flask import Flask, request, jsonify
# from google.cloud import vision, storage, firestore
# import os

# app = Flask(__name__)

# # Initialize Google Cloud Clients
# vision_client = vision.ImageAnnotatorClient()
# storage_client = storage.Client()
# firestore_client = firestore.Client()

# # Endpoint to handle file uploads
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400

#     file = request.files['file']
#     # Save file to Cloud Storage
#     bucket = storage_client.bucket('your-bucket-name')
#     blob = bucket.blob(file.filename)
#     blob.upload_from_file(file)

#     # Perform OCR using Vision API
#     image = vision.Image(source=vision.ImageSource(gcs_image_uri=f'gs://your-bucket-name/{file.filename}'))
#     response = vision_client.text_detection(image=image)

#     # Extract text
#     text = response.text_annotations[0].description if response.text_annotations else ''

#     # Perform post-processing using a custom UTRNet model (pseudo-code)
#     # processed_text = post_process_with_utrnet(text)

#     # Store results in Firestore
#     doc_ref = firestore_client.collection('documents').add({
#         'filename': file.filename,
#         'text': text
#     })

#     return jsonify({'text': text}), 200

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)


import os
from flask import Flask, request, jsonify
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import torch
from google.cloud import vision, storage, firestore
from model import Model
from read import text_recognizer
from utils import CTCLabelConverter
from ultralytics import YOLO

app = Flask(__name__)

# Load models
file = open("UrduGlyphs.txt", "r", encoding="utf-8")
content = file.readlines()
content = ''.join([str(elem).strip('\n') for elem in content])
content = content + " "
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
converter = CTCLabelConverter(content)
recognition_model = Model(num_class=len(converter.character), device=device)
recognition_model.load_state_dict(torch.load("best_norm_ED.pth", map_location=device))
recognition_model.eval()

detection_model = YOLO("yolov8m_UrduDoc.pt")

# Initialize Google Cloud clients
vision_client = vision.ImageAnnotatorClient()
storage_client = storage.Client()
firestore_client = firestore.Client()

# Define directories for processing
pngs_root_folder = 'pdf_images'
text_root_folder = 'extracted_texts'

# Ensure output folders exist
os.makedirs(pngs_root_folder, exist_ok=True)
os.makedirs(text_root_folder, exist_ok=True)

def preprocess_image(img):
    img = img.convert('L')  # Convert to grayscale
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)  # Increase contrast
    img = img.filter(ImageFilter.MedianFilter())  # Reduce noise
    return img

def predict(input):
    detection_results = detection_model.predict(source=input, conf=0.2, imgsz=1280, save=False, nms=True, device=device)
    bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
    bounding_boxes.sort(key=lambda x: x[1])  # Sort bounding boxes

    cropped_images = [input.crop(box) for box in bounding_boxes]

    texts = [text_recognizer(img, recognition_model, converter, device) for img in cropped_images]
    return "\n".join(texts)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    # Save file to Cloud Storage
    bucket_name = 'your-bucket-name'
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file.filename)
    blob.upload_from_file(file)

    # Convert PDF to images
    pdf_file_path = file.filename
    images = convert_from_path(pdf_file_path, dpi=300, fmt='png', paths_only=True, output_folder=pngs_root_folder)

    all_text = ''
    for img_path in images:
        img = preprocess_image(Image.open(img_path))
        _, text = predict(img)
        all_text += text + '\n\n'

    # Save results to Firestore
    doc_ref = firestore_client.collection('documents').add({
        'filename': file.filename,
        'text': all_text
    })

    return jsonify({'text': all_text}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
