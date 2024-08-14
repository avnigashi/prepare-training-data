import sys
import os
import cv2
import base64
import requests
import zipfile
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QListWidget, QLabel, QLineEdit, QRadioButton, 
                             QProgressBar, QFileDialog, QMessageBox, QGridLayout, QCheckBox,
                             QTabWidget, QScrollArea, QListWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor

class ImageProcessor(QThread):
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    image_processed = pyqtSignal(str, bool, str)
    processing_finished = pyqtSignal(dict)

    def __init__(self, folders, output_dir, min_width, min_height, generate_captions, caption_limit, api_params, ai_validation, validation_prompt):
        super().__init__()
        self.folders = folders
        self.output_dir = output_dir
        self.min_width = min_width
        self.min_height = min_height
        self.generate_captions = generate_captions
        self.caption_limit = caption_limit
        self.api_params = api_params
        self.ai_validation = ai_validation
        self.validation_prompt = validation_prompt

    def run(self):
        total_files = sum(len([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]) for folder in self.folders)
        processed_files = 0
        caption_count = 0
        stats = {
            'total_images': total_files,
            'faces_found': 0,
            'no_faces': 0,
            'small_images': 0,
            'failed_validation': 0,
            'processed_successfully': 0
        }

        # Load the pre-trained face detector model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        for folder in self.folders:
            for image_file in os.listdir(folder):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.status_update.emit(f"Processing {image_file}...")
                    
                    image_path = os.path.join(folder, image_file)
                    image = cv2.imread(image_path)
                    
                    # Check image dimensions
                    height, width = image.shape[:2]
                    if width < self.min_width or height < self.min_height:
                        print(f"Ignoring {image_file} due to small size ({width}x{height})")
                        stats['small_images'] += 1
                        self.image_processed.emit(image_file, False, "Small image")
                        processed_files += 1
                        self.progress_update.emit(int((processed_files / total_files) * 100))
                        continue

                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces in the image
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    
                    if len(faces) > 0:
                        stats['faces_found'] += 1
                        # Crop the first detected face
                        (x, y, w, h) = faces[0]
                        cropped_face = image[y:y+h, x:x+w]
                        cropped_path = os.path.join(self.output_dir, image_file)
                        
                        # AI Validation
                        if self.ai_validation:
                            if not self.validate_image(cropped_face):
                                print(f"Image {image_file} did not pass AI validation")
                                stats['failed_validation'] += 1
                                self.image_processed.emit(image_file, False, "Failed AI validation")
                                processed_files += 1
                                self.progress_update.emit(int((processed_files / total_files) * 100))
                                continue

                        # Save the cropped face image
                        cv2.imwrite(cropped_path, cropped_face)
                        print(f"Cropped face saved to {cropped_path}")

                        # Generate caption if enabled
                        if self.generate_captions and (self.caption_limit is None or caption_count < self.caption_limit):
                            self.generate_caption(cropped_path)
                            caption_count += 1

                        stats['processed_successfully'] += 1
                        self.image_processed.emit(image_file, True, "Processed successfully")
                    else:
                        print(f"No face detected in {image_file}")
                        stats['no_faces'] += 1
                        self.image_processed.emit(image_file, False, "No face detected")

                    processed_files += 1
                    self.progress_update.emit(int((processed_files / total_files) * 100))

        if self.generate_captions:
            self.prepare_training_data()

        self.status_update.emit("Processing complete!")
        self.processing_finished.emit(stats)

    def validate_image(self, image):
        # Convert image to base64
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Prepare the API request
        url = "http://localhost:11434/api/generate"  # Adjust if necessary
        payload = {
            "model": "llava",
            "prompt": f"{self.validation_prompt}\n[IMAGE]{image_base64}[/IMAGE]",
            "stream": False
        }

        # Send the request
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()['response'].lower()
            return "yes" in result or "true" in result
        else:
            print(f"Error in AI validation: {response.status_code} - {response.text}")
            return False

    def generate_caption(self, image_path):
        with open(image_path, "rb") as file:
            image_base64 = base64.b64encode(file.read()).decode('utf-8')
        
        caption = self.analyze_video_content(image_base64)
        print(f"Generated Caption for {os.path.basename(image_path)}: {caption}")

        # Save the caption to a text file
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        caption_dir = os.path.join(self.output_dir, "captions")
        os.makedirs(caption_dir, exist_ok=True)
        caption_path = os.path.join(caption_dir, f"{base_name}.txt")
        with open(caption_path, "w") as caption_file:
            caption_file.write(caption)

    def analyze_video_content(self, image_base64):
        url = self.api_params['url']

        payload = {
            "model": self.api_params['model'],
            "prompt": self.api_params['prompt'],
            "images": [image_base64],
            "options": {
                "temperature": float(self.api_params['temperature']),
                "max_tokens": int(self.api_params['max_tokens']),
                "top_p": float(self.api_params['top_p']),
                "frequency_penalty": float(self.api_params['frequency_penalty']),
                "presence_penalty": float(self.api_params['presence_penalty'])
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

    def prepare_training_data(self):
        image_dir = self.output_dir
        caption_dir = os.path.join(self.output_dir, "captions")
        output_zip = os.path.join(self.output_dir, "training_data.zip")

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

        self.status_update.emit(f"Training data prepared and saved to {output_zip}")

class FaceCroppingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Cropping and Captioning App")
        self.setGeometry(100, 100, 1000, 800)  # Increased window size
        self.folders = []
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create tabs
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)

        # Main tab
        main_tab = QWidget()
        tab_widget.addTab(main_tab, "Main")
        layout = QVBoxLayout(main_tab)

        # Folder list
        self.folder_list = QListWidget()
        layout.addWidget(QLabel("Folders to Process:"))
        layout.addWidget(self.folder_list)

        # Buttons
        button_layout = QHBoxLayout()
        self.add_folder_btn = QPushButton("Add Folder")
        self.remove_folder_btn = QPushButton("Remove Folder")
        self.add_folder_btn.clicked.connect(self.add_folder)
        self.remove_folder_btn.clicked.connect(self.remove_folder)
        button_layout.addWidget(self.add_folder_btn)
        button_layout.addWidget(self.remove_folder_btn)
        layout.addLayout(button_layout)

        # Output directory
        output_layout = QHBoxLayout()
        self.output_dir = QLineEdit()
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(QLabel("Output Directory:"))
        output_layout.addWidget(self.output_dir)
        output_layout.addWidget(self.browse_btn)
        layout.addLayout(output_layout)

        # Minimum size inputs
        size_layout = QGridLayout()
        self.min_width = QLineEdit("0")
        self.min_height = QLineEdit("0")
        size_layout.addWidget(QLabel("Minimum Width:"), 0, 0)
        size_layout.addWidget(self.min_width, 0, 1)
        size_layout.addWidget(QLabel("Minimum Height:"), 1, 0)
        size_layout.addWidget(self.min_height, 1, 1)
        layout.addLayout(size_layout)

        # Caption generation options
        caption_layout = QHBoxLayout()
        self.generate_captions = QCheckBox("Generate Captions")
        self.caption_limit = QLineEdit("30")
        caption_layout.addWidget(self.generate_captions)
        caption_layout.addWidget(QLabel("Caption Limit:"))
        caption_layout.addWidget(self.caption_limit)
        layout.addLayout(caption_layout)

        # AI Validation
        ai_validation_layout = QVBoxLayout()
        self.ai_validation = QCheckBox("AI Validation")
        self.validation_prompt = QLineEdit("Does this image contain a clear, front-facing human face? Answer with yes or no.")
        ai_validation_layout.addWidget(self.ai_validation)
        ai_validation_layout.addWidget(QLabel("Validation Prompt:"))
        ai_validation_layout.addWidget(self.validation_prompt)
        layout.addLayout(ai_validation_layout)

        # Processed Images List
        self.processed_images_list = QListWidget()
        layout.addWidget(QLabel("Processed Images:"))
        layout.addWidget(self.processed_images_list)

        # Summary Section
        self.summary_label = QLabel("Processing Summary:")
        layout.addWidget(self.summary_label)

        # API Parameters tab
        api_tab = QWidget()
        tab_widget.addTab(api_tab, "API Parameters")
        api_layout = QVBoxLayout(api_tab)

        # Scroll area for API parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        self.api_params = {}
        for param, default in [
            ('url', "http://localhost:11434/api/generate"),
            ('model', "llava-llama3"),
            ('prompt', "Return tags describing this picture. Use single words or short phrases separated by commas."),
            ('temperature', "0.7"),
            ('max_tokens', "1000"),
            ('top_p', "1"),
            ('frequency_penalty', "0"),
            ('presence_penalty', "0")
        ]:
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param.capitalize()}:"))
            self.api_params[param] = QLineEdit(default)
            param_layout.addWidget(self.api_params[param])
            scroll_layout.addLayout(param_layout)

        scroll.setWidget(scroll_content)
        api_layout.addWidget(scroll)

        # Process button
        self.process_btn = QPushButton("Process Images")
        self.process_btn.clicked.connect(self.start_processing)
        main_layout.addWidget(self.process_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("")
        main_layout.addWidget(self.status_label)

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Process")
        if folder:
            if folder not in self.folders:
                self.folders.append(folder)
                self.folder_list.addItem(folder)
            else:
                QMessageBox.information(self, "Duplicate Folder", "This folder is already in the list.")

    def remove_folder(self):
        current_item = self.folder_list.currentItem()
        if current_item:
            folder = current_item.text()
            self.folders.remove(folder)
            self.folder_list.takeItem(self.folder_list.row(current_item))
        else:
            QMessageBox.information(self, "No Selection", "Please select a folder to remove.")

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_dir.setText(folder)

    def start_processing(self):
        if not self.folders:
            QMessageBox.warning(self, "No Folders", "Please add at least one folder to process.")
            return
        if not self.output_dir.text():
            QMessageBox.warning(self, "No Output Directory", "Please select an output directory.")
            return

        try:
            min_width = int(self.min_width.text())
            min_height = int(self.min_height.text())
            caption_limit = int(self.caption_limit.text()) if self.generate_captions.isChecked() else None
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid integers for minimum width, height, and caption limit.")
            return

        # Clear the processed images list
        self.processed_images_list.clear()

        # Ensure the output directory exists
        os.makedirs(self.output_dir.text(), exist_ok=True)

        api_params = {param: widget.text() for param, widget in self.api_params.items()}

        self.image_processor = ImageProcessor(
            self.folders, 
            self.output_dir.text(), 
            min_width, 
            min_height,
            self.generate_captions.isChecked(),
            caption_limit,
            api_params,
            self.ai_validation.isChecked(),
            self.validation_prompt.text()
        )
        self.image_processor.progress_update.connect(self.update_progress)
        self.image_processor.status_update.connect(self.update_status)
        self.image_processor.image_processed.connect(self.update_image_list)
        self.image_processor.processing_finished.connect(self.show_summary)
        self.image_processor.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, message):
        self.status_label.setText(message)

    def update_image_list(self, image_name, success, reason):
        item = QListWidgetItem(f"{image_name} - {reason}")
        if not success:
            item.setBackground(QColor(255, 200, 200))  # Light red background for failed images
        self.processed_images_list.addItem(item)

    def show_summary(self, stats):
        summary = f"""
        Processing Summary:
        Total Images: {stats['total_images']}
        Faces Found: {stats['faces_found']}
        No Faces Detected: {stats['no_faces']}
        Small Images Skipped: {stats['small_images']}
        Failed AI Validation: {stats['failed_validation']}
        Successfully Processed: {stats['processed_successfully']}
        """
        self.summary_label.setText(summary)
        QMessageBox.information(self, "Processing Complete", summary)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FaceCroppingApp()
    ex.show()
    sys.exit(app.exec_())