# # # import torch
# # # import gradio as gr
# # # from read import text_recognizer
# # # from model import Model
# # # from utils import CTCLabelConverter
# # # from ultralytics import YOLO
# # # from PIL import ImageDraw

# # # """ vocab / character number configuration """
# # # file = open("UrduGlyphs.txt","r",encoding="utf-8")
# # # content = file.readlines()
# # # content = ''.join([str(elem).strip('\n') for elem in content])
# # # content = content+" "
# # # """ model configuration """
# # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # # converter = CTCLabelConverter(content)
# # # recognition_model = Model(num_class=len(converter.character), device=device)
# # # recognition_model = recognition_model.to(device)
# # # recognition_model.load_state_dict(torch.load("best_norm_ED.pth", map_location=device))
# # # recognition_model.eval()

# # # detection_model = YOLO("yolov8m_UrduDoc.pt")

# # # examples = ["1.jpg","2.jpg","3.jpg"]

# # # input = gr.Image(type="pil",image_mode="RGB", label="Input Image")

# # # def predict(input):
# # #     "Line Detection"
# # #     detection_results = detection_model.predict(source=input, conf=0.2, imgsz=1280, save=False, nms=True, device=device)
# # #     bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
# # #     bounding_boxes.sort(key=lambda x: x[1])
    
# # #     "Draw the bounding boxes"
# # #     draw = ImageDraw.Draw(input)
# # #     for box in bounding_boxes:
# # #         # draw rectangle outline with random color and width=5
# # #         from numpy import random
# # #         draw.rectangle(box, fill=None, outline=tuple(random.randint(0,255,3)), width=5)
    
# # #     "Crop the detected lines"
# # #     cropped_images = []
# # #     for box in bounding_boxes:
# # #         cropped_images.append(input.crop(box))
# # #     len(cropped_images)
    
# # #     "Recognize the text"
# # #     texts = []
# # #     for img in cropped_images:
# # #         texts.append(text_recognizer(img, recognition_model, converter, device))
    
# # #     "Join the text"
# # #     text = "\n".join(texts)
    
# # #     "Return the image with bounding boxes and the text"
# # #     return input,text

# # # output_image = gr.Image(type="pil",image_mode="RGB",label="Detected Lines")
# # # output_text = gr.Textbox(label="Recognized Text",interactive=True,show_copy_button=True)

# # # iface = gr.Interface(predict,
# # #                      inputs=input,
# # #                      outputs=[output_image,output_text],
# # #                      title="End-to-End Urdu OCR",
# # #                      description="Demo Web App For UTRNet\n(https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition)",
# # #                      examples=examples,
# # #                      allow_flagging="never")
# # # iface.launch()

# # import os
# # from pdf2image import convert_from_path
# # from PIL import Image, ImageEnhance, ImageFilter
# # import torch
# # from read import text_recognizer
# # from model import Model
# # from utils import CTCLabelConverter
# # from ultralytics import YOLO

# # # Load the recognition model
# # # ... same as before ...
# # """ vocab / character number configuration """
# # file = open("UrduGlyphs.txt","r",encoding="utf-8")
# # content = file.readlines()
# # content = ''.join([str(elem).strip('\n') for elem in content])
# # content = content+" "
# # """ model configuration """
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # converter = CTCLabelConverter(content)
# # recognition_model = Model(num_class=len(converter.character), device=device)
# # recognition_model = recognition_model.to(device)
# # recognition_model.load_state_dict(torch.load("best_norm_ED.pth", map_location=device))
# # recognition_model.eval()

# # detection_model = YOLO("yolov8m_UrduDoc.pt")


# # # Define directories
# # books_folder = 'C:\\Urdu_LLM\\urdu_books\\ocr model\\End-To-End-Urdu-OCR-WebApp\\books'
# # pngs_root_folder = 'C:\\Urdu_LLM\\urdu_books\\ocr model\\End-To-End-Urdu-OCR-WebApp\\pdf_images'
# # text_root_folder = 'C:\\Urdu_LLM\\urdu_books\\ocr model\\End-To-End-Urdu-OCR-WebApp\\extracted_texts'

# # # Ensure the output folders exist
# # # ... same as before ...
# # if not os.path.exists(pngs_root_folder):
# #     os.makedirs(pngs_root_folder)
# # if not os.path.exists(text_root_folder):
# #     os.makedirs(text_root_folder)

# # def preprocess_image(img):
# #     # ... same as before ...
# #     # Convert to grayscale
# #     img = img.convert('L')
# #     # Increase contrast
# #     enhancer = ImageEnhance.Contrast(img)
# #     img = enhancer.enhance(2)
# #     # Apply a filter to reduce noise
# #     img = img.filter(ImageFilter.MedianFilter())
# #     return img

# # def predict(input):
# #     "Line Detection"
# #     detection_results = detection_model.predict(source=input, conf=0.2, imgsz=1280, save=False, nms=True, device=device)
# #     bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
# #     bounding_boxes.sort(key=lambda x: x[1])
    
# #     # "Draw the bounding boxes"
# #     # draw = ImageDraw.Draw(input)
# #     # for box in bounding_boxes:
# #     #     # draw rectangle outline with random color and width=5
# #     #     from numpy import random
# #     #     draw.rectangle(box, fill=None, outline=tuple(random.randint(0,255,3)), width=5)
    
# #     "Crop the detected lines"
# #     cropped_images = []
# #     for box in bounding_boxes:
# #         cropped_images.append(input.crop(box))
# #     len(cropped_images)
    
# #     "Recognize the text"
# #     texts = []
# #     for img in cropped_images:
# #         texts.append(text_recognizer(img, recognition_model, converter, device))
    
# #     "Join the text"
# #     text = "\n".join(texts)
    
# #     "Return the image with bounding boxes and the text"
# #     return input,text

# # def process_pdf(pdf_file_path):
# #     # Extract book index from the file name
# #     pdf_filename = os.path.basename(pdf_file_path)
# #     book_index = pdf_filename.split('.')[0]
    
# #     # Create folders for images and text output
# #     book_images_folder = os.path.join(pngs_root_folder, f'book{book_index}')
# #     book_text_file = os.path.join(text_root_folder, f'book{book_index}.txt')
    
# #     if not os.path.exists(book_images_folder):
# #         os.makedirs(book_images_folder)
    
# #     # Convert PDF to images
# #     images = convert_from_path(pdf_file_path, dpi=300, output_folder=book_images_folder, fmt='png', paths_only=True)
    
# #     # Open the output text file in write mode
# #     with open(book_text_file, 'w', encoding='utf-8') as f:
# #         # Loop through all converted images
# #         for img_path in images:
# #             # Preprocess the image
# #             img = preprocess_image(Image.open(img_path))
# #             # Perform OCR on the image using UTRNet
# #             _, text = predict(img)
# #             # Write the cleaned text to the output file
# #             f.write(text)
# #             f.write('\n\n')
    
# #     print(f"Text extraction for {pdf_filename} complete. Check {book_text_file} for the results.")

# # # Process all PDF files in the 'books' folder
# # for filename in os.listdir(books_folder):
# #     if filename.endswith('.pdf'):
# #         process_pdf(os.path.join(books_folder, filename))



# # import os
# # from pdf2image import convert_from_path
# # from PIL import Image, ImageEnhance, ImageFilter
# # import torch
# # from read import text_recognizer
# # from model import Model
# # from utils import CTCLabelConverter
# # from ultralytics import YOLO
# # from multiprocessing import Pool, cpu_count

# # # Load the recognition model
# # file = open("UrduGlyphs.txt", "r", encoding="utf-8")
# # content = file.readlines()
# # content = ''.join([str(elem).strip('\n') for elem in content])
# # content = content + " "
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # converter = CTCLabelConverter(content)
# # recognition_model = Model(num_class=len(converter.character), device=device)
# # recognition_model = recognition_model.to(device)
# # recognition_model.load_state_dict(torch.load("best_norm_ED.pth", map_location=device))
# # recognition_model.eval()

# # detection_model = YOLO("yolov8m_UrduDoc.pt")

# # # Define directories
# # books_folder = 'C:\\Urdu_LLM\\urdu_books\\ocr model\\End-To-End-Urdu-OCR-WebApp\\books'
# # pngs_root_folder = 'C:\\Urdu_LLM\\urdu_books\\ocr model\\End-To-End-Urdu-OCR-WebApp\\pdf_images'
# # text_root_folder = 'C:\\Urdu_LLM\\urdu_books\\ocr model\\End-To-End-Urdu-OCR-WebApp\\extracted_texts'

# # # Ensure the output folders exist
# # if not os.path.exists(pngs_root_folder):
# #     os.makedirs(pngs_root_folder)
# # if not os.path.exists(text_root_folder):
# #     os.makedirs(text_root_folder)

# # def preprocess_image(img):
# #     # Convert to grayscale
# #     img = img.convert('L')
# #     # Increase contrast
# #     enhancer = ImageEnhance.Contrast(img)
# #     img = enhancer.enhance(2)
# #     # Apply a filter to reduce noise
# #     img = img.filter(ImageFilter.MedianFilter())
# #     return img

# # def predict(input):
# #     "Line Detection"
# #     detection_results = detection_model.predict(source=input, conf=0.2, imgsz=1280, save=False, nms=True, device=device)
# #     bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
# #     bounding_boxes.sort(key=lambda x: x[1])

# #     "Crop the detected lines"
# #     cropped_images = []
# #     for box in bounding_boxes:
# #         cropped_images.append(input.crop(box))

# #     "Recognize the text"
# #     texts = []
# #     for img in cropped_images:
# #         texts.append(text_recognizer(img, recognition_model, converter, device))

# #     "Join the text"
# #     text = "\n".join(texts)

# #     "Return the image with bounding boxes and the text"
# #     return input, text

# # def process_pdf(pdf_file_path):
# #     # Extract book index from the file name
# #     pdf_filename = os.path.basename(pdf_file_path)
# #     book_index = pdf_filename.split('.')[0]

# #     # Create folders for images and text output
# #     book_images_folder = os.path.join(pngs_root_folder, f'book{book_index}')
# #     book_text_file = os.path.join(text_root_folder, f'book{book_index}.txt')

# #     if not os.path.exists(book_images_folder):
# #         os.makedirs(book_images_folder)

# #     # Convert PDF to images
# #     images = convert_from_path(pdf_file_path, dpi=300, output_folder=book_images_folder, fmt='png', paths_only=True)

# #     # Open the output text file in write mode
# #     with open(book_text_file, 'w', encoding='utf-8') as f:
# #         # Loop through all converted images
# #         for page_number, img_path in enumerate(images, start=1):
# #             # Preprocess the image
# #             img = preprocess_image(Image.open(img_path))
# #             # Perform OCR on the image using UTRNet
# #             _, text = predict(img)
# #             # Write the cleaned text to the output file
# #             f.write(f"Page {page_number}:\n")
# #             f.write(text)
# #             f.write('\n\n')
# #             print(f"Scanned page {page_number} of {pdf_filename}")

# #     print(f"Text extraction for {pdf_filename} complete. Check {book_text_file} for the results.")

# # def process_all_pdfs(pdf_files):
# #     with Pool(processes=cpu_count()) as pool:
# #         pool.map(process_pdf, pdf_files)

# # if __name__ == '__main__':
# #     pdf_files = [os.path.join(books_folder, filename) for filename in os.listdir(books_folder) if filename.endswith('.pdf')]
# #     process_all_pdfs(pdf_files)



# import os
# from pdf2image import convert_from_path
# from PIL import Image, ImageEnhance, ImageFilter
# import torch
# from read import text_recognizer
# from model import Model
# from utils import CTCLabelConverter
# from ultralytics import YOLO

# # Load the recognition model
# file = open("UrduGlyphs.txt", "r", encoding="utf-8")
# content = file.readlines()
# content = ''.join([str(elem).strip('\n') for elem in content])
# content = content + " "
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# converter = CTCLabelConverter(content)
# recognition_model = Model(num_class=len(converter.character), device=device)
# recognition_model = recognition_model.to(device)
# recognition_model.load_state_dict(torch.load("best_norm_ED.pth", map_location=device))
# recognition_model.eval()

# detection_model = YOLO("yolov8m_UrduDoc.pt")

# # Define directories
# books_folder = 'C:\\Urdu_LLM\\urdu_books\\ocr model\\End-To-End-Urdu-OCR-WebApp\\books'
# pngs_root_folder = 'C:\\Urdu_LLM\\urdu_books\\ocr model\\End-To-End-Urdu-OCR-WebApp\\pdf_images'
# text_root_folder = 'C:\\Urdu_LLM\\urdu_books\\ocr model\\End-To-End-Urdu-OCR-WebApp\\extracted_texts'

# # Ensure the output folders exist
# if not os.path.exists(pngs_root_folder):
#     os.makedirs(pngs_root_folder)
# if not os.path.exists(text_root_folder):
#     os.makedirs(text_root_folder)

# def preprocess_image(img):
#     # Convert to grayscale
#     img = img.convert('L')
#     # Increase contrast
#     enhancer = ImageEnhance.Contrast(img)
#     img = enhancer.enhance(2)
#     # Apply a filter to reduce noise
#     img = img.filter(ImageFilter.MedianFilter())
#     return img

# def predict(input):
#     "Line Detection"
#     detection_results = detection_model.predict(source=input, conf=0.2, imgsz=1280, save=False, nms=True, device=device)
#     bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
#     bounding_boxes.sort(key=lambda x: x[1])

#     "Crop the detected lines"
#     cropped_images = []
#     for box in bounding_boxes:
#         cropped_images.append(input.crop(box))

#     "Recognize the text"
#     texts = []
#     for img in cropped_images:
#         texts.append(text_recognizer(img, recognition_model, converter, device))

#     "Join the text"
#     text = "\n".join(texts)

#     "Return the image with bounding boxes and the text"
#     return input, text

# def process_pdf(pdf_file_path, index):
#     # Extract book index from the file name
#     pdf_filename = os.path.basename(pdf_file_path)
#     book_index = index + 1  # Indexing starts from 1

#     # Create folders for images and text output
#     book_images_folder = os.path.join(pngs_root_folder, f'book33.{book_index}')
#     book_text_file = os.path.join(text_root_folder, f'book33.{book_index}.txt')

#     if not os.path.exists(book_images_folder):
#         os.makedirs(book_images_folder)

#     # Convert PDF to images
#     images = convert_from_path(pdf_file_path, dpi=300, output_folder=book_images_folder, fmt='png', paths_only=True)

#     # Open the output text file in write mode
#     with open(book_text_file, 'w', encoding='utf-8') as f:
#         # Loop through all converted images
#         for page_number, img_path in enumerate(images, start=1):
#             # Preprocess the image
#             img = preprocess_image(Image.open(img_path))
#             # Perform OCR on the image using UTRNet
#             _, text = predict(img)
#             # Write the cleaned text to the output file
#             f.write(f"Page {page_number}:\n")
#             f.write(text)
#             f.write('\n\n')
#             print(f"Scanned page {page_number} of {pdf_filename}")

#     print(f"Text extraction for {pdf_filename} complete. Check {book_text_file} for the results.")

# if __name__ == '__main__':
#     # Get the list of PDF files with their sizes
#     pdf_files_with_sizes = [(os.path.join(books_folder, filename), os.path.getsize(os.path.join(books_folder, filename)))
#                             for filename in os.listdir(books_folder) if filename.endswith('.pdf')]
#     # Sort the files by size (largest first)
#     pdf_files_with_sizes.sort(key=lambda x: x[1], reverse=True)
#     pdf_files = [file[0] for file in pdf_files_with_sizes]

#     # Process PDFs one at a time
#     for index, pdf_file in enumerate(pdf_files):
#         process_pdf(pdf_file, index)




import os
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import torch
from read import text_recognizer
from model import Model
from utils import CTCLabelConverter
from ultralytics import YOLO

# Load the recognition model
file = open("/mnt/HDFS1/llm/llmurdu/Mohd_Rashid/Urdu-OCR-for-Scanned-PDFs/UrduGlyphs.txt", "r", encoding="utf-8")
content = file.readlines()
content = ''.join([str(elem).strip('\n') for elem in content])
content = content + " "
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
converter = CTCLabelConverter(content)
recognition_model = Model(num_class=len(converter.character), device=device)
recognition_model = recognition_model.to(device)
recognition_model.load_state_dict(torch.load("/mnt/HDFS1/llm/llmurdu/Mohd_Rashid/Urdu-OCR-for-Scanned-PDFs/best_norm_ED.pth", map_location=device))
recognition_model.eval()

detection_model = YOLO("/mnt/HDFS1/llm/llmurdu/Mohd_Rashid/Urdu-OCR-for-Scanned-PDFs/yolov8m_UrduDoc.pt")

# Define directories
books_folder = '/mnt/HDFS1/llm/llmurdu/Mohd_Rashid/Urdu-OCR-for-Scanned-PDFs/books18'
pngs_root_folder = '/mnt/HDFS1/llm/llmurdu/Mohd_Rashid/Urdu-OCR-for-Scanned-PDFs/pdf_images'
text_root_folder = '/mnt/HDFS1/llm/llmurdu/Mohd_Rashid/Urdu-OCR-for-Scanned-PDFs/extracted_texts'

# Ensure the output folders exist
if not os.path.exists(pngs_root_folder):
    os.makedirs(pngs_root_folder)
if not os.path.exists(text_root_folder):
    os.makedirs(text_root_folder)

def preprocess_image(img):
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

    "Crop the detected lines"
    cropped_images = []
    for box in bounding_boxes:
        cropped_images.append(input.crop(box))

    "Recognize the text"
    texts = []
    for img in cropped_images:
        texts.append(text_recognizer(img, recognition_model, converter, device))

    "Join the text"
    text = "\n".join(texts)

    "Return the image with bounding boxes and the text"
    return input, text

def process_pdf(pdf_file_path, index, split_pages=False):
    # Extract book index from the file name
    pdf_filename = os.path.basename(pdf_file_path)
    book_index = index + 1  # Indexing starts from 1

    # Create folders for images and text output
    book_images_folder = os.path.join(pngs_root_folder, f'book.{book_index}')
    if split_pages:
        book_text_folder = os.path.join(text_root_folder, f'book.{book_index}')
        if not os.path.exists(book_text_folder):
            os.makedirs(book_text_folder)
    else:
        book_text_file = os.path.join(text_root_folder, f'book7.{book_index}.txt')

    if not os.path.exists(book_images_folder):
        os.makedirs(book_images_folder)

    # Convert PDF to images
    images = convert_from_path(pdf_file_path, dpi=300, output_folder=book_images_folder, fmt='png', paths_only=True)

    # Open the output text file in write mode if not splitting pages
    if not split_pages:
        text_file = open(book_text_file, 'w', encoding='utf-8')

    seen_lines = set()

    # Loop through all converted images
    for page_number, img_path in enumerate(images, start=1):
        # Preprocess the image
        img = preprocess_image(Image.open(img_path))
        # Perform OCR on the image using UTRNet
        _, text = predict(img)

        if split_pages:
            page_text_file = os.path.join(book_text_folder, f'page_{page_number}.txt')
            with open(page_text_file, 'w', encoding='utf-8') as page_file:
                for line in text.split('\n'):
                    if line not in seen_lines:
                        page_file.write(line + '\n')
                        seen_lines.add(line)
        else:
            text_file.write(f"Page {page_number}:\n")
            for line in text.split('\n'):
                if line not in seen_lines:
                    text_file.write(line + '\n')
                    seen_lines.add(line)
            text_file.write('\n\n')

        print(f"Scanned page {page_number} of {pdf_filename}")

    if not split_pages:
        text_file.close()

    print(f"Text extraction for {pdf_filename} complete.")

# # Uncomment the following block to print comparison data
# def print_comparison_data(before_text, after_text):
#     before_lines = before_text.split('\n')
#     after_lines = after_text.split('\n')
#     print(f"Total lines before: {len(before_lines)}")
#     print(f"Total lines after: {len(after_lines)}")
#     print(f"Duplicate lines removed: {len(before_lines) - len(after_lines)}")

if __name__ == '__main__':
    pdf_files_with_sizes = sorted(
        [(os.path.join(books_folder, filename), os.path.getsize(os.path.join(books_folder, filename)))
         for filename in os.listdir(books_folder) if filename.endswith('.pdf')],
        key=lambda x: x[1]
    )

    pdf_files = [file[0] for file in pdf_files_with_sizes]

    for index, pdf_file in enumerate(pdf_files):
        process_pdf(pdf_file, index, split_pages=False)

