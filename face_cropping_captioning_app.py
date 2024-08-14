import sys
import os
import cv2
import base64
import requests
import zipfile
import json
from PyQt5.QtGui import QPixmap, QIcon, QColor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QListWidget, QLabel, QLineEdit, QCheckBox,
                             QProgressBar, QFileDialog, QMessageBox, QGridLayout, QTabWidget, QScrollArea, QListWidgetItem, QInputDialog, QComboBox, QStatusBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

class ImageProcessor(QThread):
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    image_processed = pyqtSignal(str, bool, str)
    processing_finished = pyqtSignal(dict)

    def __init__(self, folders, output_dir, min_width, min_height, generate_captions, caption_limit, api_params, ai_validation, allowed_file_types):
        super().__init__()
        self.folders = folders
        self.output_dir = output_dir
        self.min_width = min_width
        self.min_height = min_height
        self.generate_captions = generate_captions
        self.caption_limit = caption_limit
        self.api_params = api_params
        self.ai_validation = ai_validation
        self.allowed_file_types = allowed_file_types

    def run(self):
        total_files = sum(len([f for f in os.listdir(folder) if f.lower().endswith(tuple(self.allowed_file_types))]) for folder in self.folders)
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
                if image_file.lower().endswith(tuple(self.allowed_file_types)):
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
        url = self.api_params['url'] + self.api_params['generate_uri']
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

        caption = self.analyze_image_content(image_base64)
        print(f"Generated Caption for {os.path.basename(image_path)}: {caption}")

        # Save the caption to a text file
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        caption_dir = os.path.join(self.output_dir, "captions")
        os.makedirs(caption_dir, exist_ok=True)
        caption_path = os.path.join(caption_dir, f"{base_name}.txt")
        with open(caption_path, "w") as caption_file:
            caption_file.write(caption)

    def analyze_image_content(self, image_base64):
        url = self.api_params['url'] + self.api_params['generate_uri']

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
                if image_file.lower().endswith(tuple(self.allowed_file_types)):
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
        self.setGeometry(100, 100, 1200, 800)

        #icon
        icon_path = os.path.join(os.path.dirname(__file__), 'assets', 'icon.png')
        self.setWindowIcon(QIcon(icon_path))

        self.folders = []
        self.allowed_file_types = ['.png', '.jpg', '.jpeg', '.webp']
        self.image_processor = None  # To track the running process
        self.processing = False  # To track whether processing is active
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)

        # Main Tab
        main_tab = QWidget()
        tab_widget.addTab(main_tab, "Main")
        layout = QVBoxLayout(main_tab)

        self.folder_list = QListWidget()
        self.folder_list.itemDoubleClicked.connect(self.edit_folder)
        layout.addWidget(QLabel("Folders to Process:"))
        layout.addWidget(self.folder_list)

        button_layout = QHBoxLayout()
        self.add_folder_btn = QPushButton("Add Folder")
        self.remove_folder_btn = QPushButton("Remove Folder")
        self.import_settings_btn = QPushButton("Import Settings")
        self.export_settings_btn = QPushButton("Export Settings")
        self.add_folder_btn.clicked.connect(self.add_folder)
        self.remove_folder_btn.clicked.connect(self.remove_folder)
        self.import_settings_btn.clicked.connect(self.import_settings)
        self.export_settings_btn.clicked.connect(self.export_settings)
        button_layout.addWidget(self.add_folder_btn)
        button_layout.addWidget(self.remove_folder_btn)
        layout.addLayout(button_layout)

        output_layout = QHBoxLayout()
        self.output_dir = QLineEdit()
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(QLabel("Output Directory:"))
        output_layout.addWidget(self.output_dir)
        output_layout.addWidget(self.browse_btn)
        layout.addLayout(output_layout)

        size_layout = QGridLayout()
        self.min_width = QLineEdit("0")
        self.min_height = QLineEdit("0")
        size_layout.addWidget(QLabel("Minimum Width:"), 0, 0)
        size_layout.addWidget(self.min_width, 0, 1)
        size_layout.addWidget(QLabel("Minimum Height:"), 1, 0)
        size_layout.addWidget(self.min_height, 1, 1)
        layout.addLayout(size_layout)

        caption_layout = QHBoxLayout()
        self.generate_captions = QCheckBox("Generate Captions")
        self.caption_limit = QLineEdit("30")
        caption_layout.addWidget(self.generate_captions)
        caption_layout.addWidget(QLabel("Caption Limit:"))
        caption_layout.addWidget(self.caption_limit)
        layout.addLayout(caption_layout)

        ai_validation_layout = QVBoxLayout()
        self.ai_validation = QCheckBox("AI Validation")
        ai_validation_layout.addWidget(self.ai_validation)
        layout.addLayout(ai_validation_layout)

        allowed_file_types_layout = QHBoxLayout()
        self.allowed_file_types_input = QLineEdit(", ".join(self.allowed_file_types))
        self.allowed_file_types_input.textChanged.connect(self.update_allowed_file_types)
        allowed_file_types_layout.addWidget(QLabel("Allowed File Types:"))
        allowed_file_types_layout.addWidget(self.allowed_file_types_input)
        layout.addLayout(allowed_file_types_layout)

        self.processed_images_list = QListWidget()
        layout.addWidget(QLabel("Processed Images:"))
        layout.addWidget(self.processed_images_list)

        self.summary_label = QLabel("Processing Summary:")
        layout.addWidget(self.summary_label)

        # API Parameters Tab
        api_tab = QWidget()
        tab_widget.addTab(api_tab, "API Parameters")
        api_layout = QVBoxLayout(api_tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # URL and Model at the top
        url_layout = QHBoxLayout()
        self.url_label = QLabel("Base URL:")
        self.url_input = QLineEdit("http://localhost:11434")
        self.url_refresh_btn = QPushButton("Refresh")
        self.url_refresh_btn.clicked.connect(self.refresh_url_status)
        url_layout.addWidget(self.url_label)
        url_layout.addWidget(self.url_input)
        url_layout.addWidget(self.url_refresh_btn)
        scroll_layout.addLayout(url_layout)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_select = QComboBox()
        self.model_refresh_btn = QPushButton("Refresh")
        self.model_refresh_btn.clicked.connect(self.populate_model_select)
        model_layout.addWidget(self.model_select)
        model_layout.addWidget(self.model_refresh_btn)
        scroll_layout.addLayout(model_layout)

        # Other API Parameters
        self.api_params = {}
        for param, default in [
            ('generate_uri', "/api/generate"),
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

        # AI Sandbox Tab
        ai_sandbox_tab = QWidget()
        tab_widget.addTab(ai_sandbox_tab, "AI Sandbox")
        sandbox_layout = QVBoxLayout(ai_sandbox_tab)

        self.sandbox_image_label = QLabel("Upload an image for AI interaction:")
        self.sandbox_image_path = QLineEdit()
        self.sandbox_browse_btn = QPushButton("Browse")
        self.sandbox_browse_btn.clicked.connect(self.browse_sandbox_image)
        self.sandbox_prompt = QLineEdit("Describe the image.")
        self.sandbox_temp = QLineEdit("0.7")
        self.sandbox_max_tokens = QLineEdit("1000")
        self.sandbox_submit_btn = QPushButton("Submit to AI")
        self.sandbox_response_label = QLabel("AI Response:")

        sandbox_layout.addWidget(self.sandbox_image_label)
        sandbox_layout.addWidget(self.sandbox_image_path)
        sandbox_layout.addWidget(self.sandbox_browse_btn)
        sandbox_layout.addWidget(QLabel("Prompt:"))
        sandbox_layout.addWidget(self.sandbox_prompt)
        sandbox_layout.addWidget(QLabel("Temperature:"))
        sandbox_layout.addWidget(self.sandbox_temp)
        sandbox_layout.addWidget(QLabel("Max Tokens:"))
        sandbox_layout.addWidget(self.sandbox_max_tokens)
        sandbox_layout.addWidget(self.sandbox_submit_btn)
        sandbox_layout.addWidget(self.sandbox_response_label)

        self.sandbox_submit_btn.clicked.connect(self.submit_sandbox)

        # Settings Tab
        settings_tab = QWidget()
        tab_widget.addTab(settings_tab, "Settings")
        settings_layout = QVBoxLayout(settings_tab)

        # Import/Export Settings Buttons
        settings_button_layout = QHBoxLayout()
        self.import_settings_btn = QPushButton("Import Settings")
        self.import_settings_btn.setMinimumHeight(40)
        self.export_settings_btn = QPushButton("Export Settings")
        self.export_settings_btn.setMinimumHeight(40)
        self.import_settings_btn.clicked.connect(self.import_settings)
        self.export_settings_btn.clicked.connect(self.export_settings)
        settings_button_layout.addWidget(self.import_settings_btn)
        settings_button_layout.addWidget(self.export_settings_btn)
        settings_layout.addLayout(settings_button_layout)

        self.process_btn = QPushButton("Process Images")
        self.process_btn.clicked.connect(self.start_processing)
        main_layout.addWidget(self.process_btn)

        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        main_layout.addWidget(self.status_label)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.populate_model_select()  # Populate models on start

    def update_allowed_file_types(self):
        self.allowed_file_types = [ftype.strip() for ftype in self.allowed_file_types_input.text().split(',')]

    def edit_folder(self, item):
        text, ok = QInputDialog.getText(self, "Edit Folder Path", "Folder Path:", QLineEdit.Normal, item.text())
        if ok and text:
            item.setText(text)
            self.update_folder_image_count(item)

    def update_folder_image_count(self, item):
        folder = item.text()
        if os.path.isdir(folder):
            image_count = len([f for f in os.listdir(folder) if f.lower().endswith(tuple(self.allowed_file_types))])
            item.setText(f"{folder} ({image_count} images)")

    def import_settings(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Import Settings", "", "JSON Files (*.json)")
        if file_name:
            with open(file_name, 'r') as file:
                settings = json.load(file)
                self.folders = settings.get('folders', [])
                self.folder_list.clear()
                for folder in self.folders:
                    item = QListWidgetItem(folder)
                    self.update_folder_image_count(item)
                    self.folder_list.addItem(item)
                self.output_dir.setText(settings.get('output_dir', ''))
                self.min_width.setText(str(settings.get('min_width', 0)))
                self.min_height.setText(str(settings.get('min_height', 0)))
                self.generate_captions.setChecked(settings.get('generate_captions', False))
                self.caption_limit.setText(str(settings.get('caption_limit', 30)))
                self.ai_validation.setChecked(settings.get('ai_validation', False))
                for key, value in settings.get('api_params', {}).items():
                    if key in self.api_params:
                        self.api_params[key].setText(value)
                self.allowed_file_types_input.setText(", ".join(settings.get('allowed_file_types', ['.png', '.jpg', '.jpeg', '.webp'])))
                QMessageBox.information(self, "Settings Imported", "Settings have been successfully imported.")

    def export_settings(self):
        settings = {
            'folders': self.folders,
            'output_dir': self.output_dir.text(),
            'min_width': int(self.min_width.text()),
            'min_height': int(self.min_height.text()),
            'generate_captions': self.generate_captions.isChecked(),
            'caption_limit': int(self.caption_limit.text()),
            'ai_validation': self.ai_validation.isChecked(),
            'api_params': {key: widget.text() for key, widget in self.api_params.items()},
            'allowed_file_types': self.allowed_file_types
        }
        file_name, _ = QFileDialog.getSaveFileName(self, "Export Settings", "", "JSON Files (*.json)")
        if file_name:
            with open(file_name, 'w') as file:
                json.dump(settings, file, indent=4)
            QMessageBox.information(self, "Settings Exported", "Settings have been successfully exported.")

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Process")
        if folder:
            if folder not in self.folders:
                self.folders.append(folder)
                item = QListWidgetItem(folder)
                self.update_folder_image_count(item)
                self.folder_list.addItem(item)
            else:
                QMessageBox.information(self, "Duplicate Folder", "This folder is already in the list.")

    def remove_folder(self):
        current_item = self.folder_list.currentItem()
        if current_item:
            folder = current_item.text().split(" (")[0]
            try:
                self.folders.remove(folder)
                self.folder_list.takeItem(self.folder_list.row(current_item))
            except ValueError:
                QMessageBox.warning(self, "Folder Not Found", "The selected folder could not be found in the list.")
        else:
            QMessageBox.information(self, "No Selection", "Please select a folder to remove.")

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.output_dir.setText(folder)

    def browse_sandbox_image(self):
        image_file, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.webp)")
        if image_file:
            self.sandbox_image_path.setText(image_file)

    def submit_sandbox(self):
        image_path = self.sandbox_image_path.text()
        if not os.path.isfile(image_path):
            QMessageBox.warning(self, "No Image", "Please upload an image.")
            return
        with open(image_path, "rb") as file:
            image_base64 = base64.b64encode(file.read()).decode('utf-8')

        url = self.url_input.text() + self.api_params['generate_uri'].text()
        payload = {
            "model": self.model_select.currentText(),
            "prompt": self.sandbox_prompt.text(),
            "images": [image_base64],
            "options": {
                "temperature": float(self.sandbox_temp.text()),
                "max_tokens": int(self.sandbox_max_tokens.text()),
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0
            },
            "stream": False
        }

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            response_json = response.json()
            self.sandbox_response_label.setText(f"AI Response: {response_json.get('response', 'No response')}")
        else:
            self.sandbox_response_label.setText(f"Error: {response.status_code} - {response.text}")

    def start_processing(self):
        if not self.processing:
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
            api_params['url'] = self.url_input.text()
            api_params['model'] = self.model_select.currentText()

            self.image_processor = ImageProcessor(
                self.folders,
                self.output_dir.text(),
                min_width,
                min_height,
                self.generate_captions.isChecked(),
                caption_limit,
                api_params,
                self.ai_validation.isChecked(),
                self.allowed_file_types
            )
            self.image_processor.progress_update.connect(self.update_progress)
            self.image_processor.status_update.connect(self.update_status)
            self.image_processor.image_processed.connect(self.update_image_list)
            self.image_processor.processing_finished.connect(self.show_summary)
            self.image_processor.start()

            # Switch button to "Cancel" mode
            self.process_btn.setText("Cancel Processing")
            self.processing = True
            self.process_btn.clicked.disconnect()
            self.process_btn.clicked.connect(self.cancel_processing)

            # Update status bar
            self.status_bar.showMessage(f"Running:🟢 | Ollama: 🟢 | Model: {self.model_select.currentText()}")

        else:
            self.cancel_processing()

    def cancel_processing(self):
        if self.processing and self.image_processor.isRunning():
            reply = QMessageBox.question(self, 'Cancel Processing',
                                         "Are you sure you want to cancel the processing?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.image_processor.terminate()
                self.update_status("Processing canceled!")
                self.processing = False
                self.process_btn.setText("Process Images")
                self.process_btn.clicked.disconnect()
                self.process_btn.clicked.connect(self.start_processing)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, message):
        self.status_label.setText(message)

    def update_image_list(self, image_name, success, reason):
        item = QListWidgetItem(f"{image_name} - {reason}")
        if not success:
            item.setBackground(QColor(255, 200, 200))
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
        self.processing = False
        self.process_btn.setText("Process Images")
        self.process_btn.clicked.disconnect()
        self.process_btn.clicked.connect(self.start_processing)

    def populate_model_select(self):
        try:
            response = requests.get(self.url_input.text() + "/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                self.model_select.clear()  # Clear existing items
                preselected_model = None
                for model in models:
                    self.model_select.addItem(model['name'])
                    if "clip" in model['details'].get('families', []):
                        preselected_model = model['name']
                        break
                if preselected_model:
                    self.model_select.setCurrentText(preselected_model)
                    model_size_mb = models[0]['size'] / (1024 * 1024)
                    self.status_bar.showMessage(f"Running:🟢 | Ollama: 🟢 | Model: {preselected_model} ({model_size_mb:.2f} MB)")
            else:
                self.model_select.clear()
                QMessageBox.warning(self, "Error", f"Could not load models: {response.status_code}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not load models: {str(e)}")

    def refresh_url_status(self):
        try:
            response = requests.get(self.url_input.text())
            if response.status_code == 200:
                self.url_label.setStyleSheet("color: green;")
                self.status_bar.showMessage(f"Running:🟢 | Ollama: 🟢 | Model: {self.model_select.currentText()}")
            else:
                self.url_label.setStyleSheet("color: red;")
                self.status_bar.showMessage(f"Running:🔴 | Ollama: 🔴 | Failed to connect to {self.url_input.text()}")
        except Exception as e:
            self.url_label.setStyleSheet("color: red;")
            self.status_bar.showMessage(f"Running:🔴 | Ollama: 🔴 | Could not reach the URL: {str(e)}")
            QMessageBox.warning(self, "Error", f"Could not reach the URL: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FaceCroppingApp()
    ex.show()
    sys.exit(app.exec_())
