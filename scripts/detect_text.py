# from google.cloud import vision
# from google.cloud import storage

# def detect_text_regions(image_path, bucket_name, blob_name):
#     client = vision.ImageAnnotatorClient()

#     # Upload image to Cloud Storage
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(blob_name)
#     blob.upload_from_filename(image_path)

#     # Perform document text detection
#     image = vision.Image()
#     image.source.image_uri = f"gs://{bucket_name}/{blob_name}"
#     response = client.document_text_detection(image=image)

#     # Parse response for text regions
#     text_regions = []
#     for page in response.full_text_annotation.pages:
#         for block in page.blocks:
#             text_region = {
#                 'bounding_box': [(vertex.x, vertex.y) for vertex in block.bounding_box.vertices],
#                 'text': block.text,
#                 'confidence': block.confidence
#             }
#             text_regions.append(text_region)

#     return text_regions


from google.cloud import vision
from google.cloud import storage
import os

def detect_text_regions(image_path, bucket_name, blob_name):
    # Initialize Google Cloud Vision and Storage clients
    client = vision.ImageAnnotatorClient()
    storage_client = storage.Client()

    # Upload image to Cloud Storage
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(image_path)

    # Perform document text detection using Vision API
    image = vision.Image()
    image.source.image_uri = f"gs://{bucket_name}/{blob_name}"
    response = client.document_text_detection(image=image)

    # Parse response to extract text regions and their bounding boxes
    text_regions = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            text_region = {
                'bounding_box': [(vertex.x, vertex.y) for vertex in block.bounding_box.vertices],
                'text': block.text,
                'confidence': block.confidence
            }
            text_regions.append(text_region)

    return text_regions

# Example usage
if __name__ == '__main__':
    image_path = 'C:\\Users\\Rashid\\Urdu-OCR-Model\\Urdu_Model_Final_Demo\\urdu-ocr\\Urdu_Model_Final_Demo\\urdu-ocr\\pdf_images\\bookIqbal-ek-Mard-Aafaqi\\982a8175-7aa3-4f88-8e32-ddffe0196fbe-07.png'
    bucket_name = 'urdu-ocr-model-bucket'
    blob_name = 'image.png'
    text_regions = detect_text_regions(image_path, bucket_name, blob_name)
    print(text_regions)
