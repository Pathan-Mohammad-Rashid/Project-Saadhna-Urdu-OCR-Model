def post_process_text(urdu_text, vision_text):
    # Simple example: use higher confidence text or combine both
    if urdu_text['confidence'] > vision_text['confidence']:
        return urdu_text['text']
    else:
        return vision_text['text']

# Example of combining text from both sources
combined_texts = []
for region in text_regions:
    urdu_text = recognize_urdu_text(Image.open(image_path).crop(region['bounding_box']))
    vision_text = region['text']
    combined_text = post_process_text({'text': urdu_text, 'confidence': 0.9}, {'text': vision_text, 'confidence': region['confidence']})
    combined_texts.append(combined_text)

# Output the combined text
final_output = '\n'.join(combined_texts)
print(final_output)
