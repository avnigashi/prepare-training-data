import cv2
import os

def crop_faces(image_dir, cropped_dir):
    # Ensure the cropped directory exists
    os.makedirs(cropped_dir, exist_ok=True)
    
    # Load the pre-trained face detector model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                # Crop the first detected face
                (x, y, w, h) = faces[0]
                cropped_face = image[y:y+h, x:x+w]
                cropped_path = os.path.join(cropped_dir, image_file)
                
                # Save the cropped face image
                cv2.imwrite(cropped_path, cropped_face)
                print(f"Cropped face saved to {cropped_path}")
            else:
                print(f"No face detected in {image_file}")

# Example usage
image_directory = "./sample/images/dua"
cropped_directory = "./sample/images/dua_cropped"

# Crop faces in all images in the folder
crop_faces(image_directory, cropped_directory)

