import os
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import torch
from read import text_recognizer
from model import Model
from utils import CTCLabelConverter
from ultralytics import YOLO

# Load the recognition model
# ... same as before ...
""" vocab / character number configuration """
file = open("C:\\Users\\Rashid\\Urdu-OCR-Model\\Urdu_Model_Final_Demo\\urdu-ocr\\UrduGlyphs.txt","r",encoding="utf-8")
content = file.readlines()
content = ''.join([str(elem).strip('\n') for elem in content])
content = content+" "
""" model configuration """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
converter = CTCLabelConverter(content)
recognition_model = Model(num_class=len(converter.character), device=device)
recognition_model = recognition_model.to(device)
recognition_model.load_state_dict(torch.load("C:\\Users\\Rashid\\Urdu-OCR-Model\\Urdu_Model_Final_Demo\\urdu-ocr\\best_norm_ED.pth", map_location=device))
recognition_model.eval()

detection_model = YOLO("C:\\Users\\Rashid\\Urdu-OCR-Model\\Urdu_Model_Final_Demo\\urdu-ocr\\yolov8m_UrduDoc.pt")


# Define directories
books_folder = 'C:\\Users\\Rashid\\Urdu-OCR-Model\\Urdu_Model_Final_Demo\\urdu-ocr\\books'
pngs_root_folder = 'C:\\Users\\Rashid\\Urdu-OCR-Model\\Urdu_Model_Final_Demo\\urdu-ocr\\pdf_images'
text_root_folder = 'C:\\Users\\Rashid\\Urdu-OCR-Model\\Urdu_Model_Final_Demo\\urdu-ocr\\extracted_texts'

# Ensure the output folders exist
# ... same as before ...
if not os.path.exists(pngs_root_folder):
    os.makedirs(pngs_root_folder)
if not os.path.exists(text_root_folder):
    os.makedirs(text_root_folder)

def preprocess_image(img):
    # ... same as before ...
    # Convert to grayscale
    img = img.convert('L')
    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    # Apply a filter to reduce noise
    img = img.filter(ImageFilter.MedianFilter())
    return img

def predict(input):
    "Line Detection"
    detection_results = detection_model.predict(source=input, conf=0.2, imgsz=1280, save=False, nms=True, device=device)
    bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
    bounding_boxes.sort(key=lambda x: x[1])
    
    # "Draw the bounding boxes"
    # draw = ImageDraw.Draw(input)
    # for box in bounding_boxes:
    #     # draw rectangle outline with random color and width=5
    #     from numpy import random
    #     draw.rectangle(box, fill=None, outline=tuple(random.randint(0,255,3)), width=5)
    
    "Crop the detected lines"
    cropped_images = []
    for box in bounding_boxes:
        cropped_images.append(input.crop(box))
    len(cropped_images)
    
    "Recognize the text"
    texts = []
    for img in cropped_images:
        texts.append(text_recognizer(img, recognition_model, converter, device))
    
    "Join the text"
    text = "\n".join(texts)
    
    "Return the image with bounding boxes and the text"
    return input,text

def process_pdf(pdf_file_path):
    # Extract book index from the file name
    pdf_filename = os.path.basename(pdf_file_path)
    book_index = pdf_filename.split('.')[0]
    
    # Create folders for images and text output
    book_images_folder = os.path.join(pngs_root_folder, f'book{book_index}')
    book_text_file = os.path.join(text_root_folder, f'book{book_index}.txt')
    
    if not os.path.exists(book_images_folder):
        os.makedirs(book_images_folder)
    
    # Convert PDF to images
    images = convert_from_path(pdf_file_path, dpi=300, output_folder=book_images_folder, fmt='png', paths_only=True)
    
    # Open the output text file in write mode
    with open(book_text_file, 'w', encoding='utf-8') as f:
        # Loop through all converted images
        for img_path in images:
            # Preprocess the image
            img = preprocess_image(Image.open(img_path))
            # Perform OCR on the image using UTRNet
            _, text = predict(img)
            # Write the cleaned text to the output file
            f.write(text)
            f.write('\n\n')
    
    print(f"Text extraction for {pdf_filename} complete. Check {book_text_file} for the results.")

# Process all PDF files in the 'books' folder
for filename in os.listdir(books_folder):
    if filename.endswith('.pdf'):
        process_pdf(os.path.join(books_folder, filename))