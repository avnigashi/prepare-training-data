import base64
import requests
import os
import zipfile

def analyze_video_content(image_base64):
    url = "http://host.docker.internal:11434/api/generate"

    payload = {
        "model": "llava-llama3",
        "prompt":  "Return tags describing this picture. Use single words or short phrases separated by commas.",
        "images": [image_base64],
        "options": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        },
        "stream": False
    }

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        response_json = response.json()
        print(f"Response JSON: {response_json}")  # Debug statement
        return response_json.get('response', 'No caption generated')
    else:
        print(f"Error: {response.status_code} - {response.text}")  # Debug statement
        return f"Error: {response.status_code} - {response.text}"

def generate_captions_for_folder(image_dir, caption_dir, limit=None):
    processed_count = 0
    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_path = os.path.join(image_dir, image_file)
            
            # Read and encode image to Base64
            with open(image_path, "rb") as file:
                image_base64 = base64.b64encode(file.read()).decode('utf-8')
            
            # Generate caption
            caption = analyze_video_content(image_base64)
            print(f"Generated Caption for {image_file}: {caption}")

            # Save the caption to a text file
            base_name = os.path.splitext(image_file)[0]
            caption_path = os.path.join(caption_dir, f"{base_name}.txt")
            with open(caption_path, "w") as caption_file:
                caption_file.write(caption)
            
            processed_count += 1
            if limit and processed_count >= limit:
                break

def prepare_training_data(image_dir, caption_dir, output_zip):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for image_file in os.listdir(image_dir):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                base_name = os.path.splitext(image_file)[0]
                caption_file = f"{base_name}.txt"
                
                if caption_file in os.listdir(caption_dir):
                    zipf.write(os.path.join(image_dir, image_file), image_file)
                    zipf.write(os.path.join(caption_dir, caption_file), caption_file)
                else:
                    print(f"Warning: Caption file for {image_file} not found.")

# Example usage
image_directory = "./sample/images/dua_cropped"
caption_directory = "./sample/images/dua_cropped_captions"
limit = 30  # Set the limit for the number of images to process

# Ensure the caption directory exists
os.makedirs(caption_directory, exist_ok=True)

# Generate captions for images in the folder with a limit
generate_captions_for_folder(image_directory, caption_directory, limit)

# Prepare the training data
output_zipfile = "training_data_dua_face.zip"
prepare_training_data(image_directory, caption_directory, output_zipfile)
