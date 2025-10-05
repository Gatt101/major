# 🧠 Alzheimer's Classification Web Application

A Flask-based web application that uses a trained YOLO model to classify brain scans for Alzheimer's disease stages.

## 🚀 Features

- **AI-Powered Classification**: Uses your trained YOLO model for accurate predictions
- **Responsive Web Interface**: Modern, user-friendly design that works on all devices
- **Real-time Processing**: Instant image analysis and results display
- **Multiple Class Support**: Classifies into 4 categories:
  - Non Demented
  - Very Mild Demented
  - Mild Demented
  - Moderate Demented
- **Drag & Drop Upload**: Easy image uploading with drag and drop support
- **Confidence Scores**: Shows prediction confidence for all classes

## 📋 Prerequisites

- Python 3.10 or higher
- uv package manager
- Trained YOLO model (`best.pt` in the `models/` directory)

## 🛠️ Installation

1. **Clone or navigate to the project directory**
2. **Install dependencies using uv**:
   ```bash
   uv sync
   ```

## 🎯 Usage

1. **Start the Flask application**:
   ```bash
   uv run python main.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Upload an image**:
   - Click "Choose Image File" or drag and drop an image
   - Supported formats: JPG, PNG, GIF, BMP, TIFF (Max: 16MB)

4. **View results**:
   - The AI model will analyze the image
   - Results show the predicted class and confidence scores
   - All class probabilities are displayed for transparency

## 📁 Project Structure

```
CarryBag/
├── main.py              # Flask application
├── pyproject.toml       # Project dependencies
├── models/
│   └── best.pt         # Trained YOLO model
├── templates/
│   └── index.html      # Web interface
└── README.md           # This file
```

## 🔧 Configuration

### Model Classes
The application is configured for 4 Alzheimer's classes:
- Class 0: Very Mild Demented
- Class 1: Non Demented
- Class 2: Moderate Demented
- Class 3: Mild Demented

To modify these classes, edit the `ALZHEIMER_CLASSES` dictionary in `main.py`.

### Model Path
The application expects the YOLO model at `models/best.pt`. If your model is in a different location, update the `model_path` variable in `main.py`.

## 🌐 API Endpoints

- `GET /` - Main web interface
- `POST /predict` - Upload image and get prediction
- `GET /classes` - Get available classification classes

## 📊 Example API Response

```json
{
  "success": true,
  "predictions": [
    {
      "class_name": "Non Demented",
      "class_index": 1,
      "confidence": 0.85,
      "all_confidences": {
        "Very Mild Demented": 0.05,
        "Non Demented": 0.85,
        "Moderate Demented": 0.02,
        "Mild Demented": 0.08
      }
    }
  ],
  "image": "base64_encoded_image",
  "filename": "brain_scan.jpg"
}
```

## 🔒 Security Features

- File type validation
- File size limits (16MB max)
- Input sanitization
- Error handling

## 🚨 Troubleshooting

### Common Issues

1. **Model not found**: Ensure `best.pt` is in the `models/` directory
2. **Import errors**: Make sure all dependencies are installed with `uv sync`
3. **File upload fails**: Check file size (must be < 16MB) and format
4. **Poor predictions**: Ensure input images match your training data format

### Error Messages

- "No file uploaded" - Select an image file before submitting
- "Invalid file type" - Use supported image formats (JPG, PNG, etc.)
- "Failed to process image" - Image format issue or model error

## 📝 Notes

- This is a diagnostic tool for educational/research purposes
- Always consult medical professionals for actual diagnosis
- The model's accuracy depends on training data quality
- Predictions should be validated by domain experts

## 🔄 Updates

To update dependencies:
```bash
uv sync --upgrade
```

## 📞 Support

If you encounter issues:
1. Check the console output for error messages
2. Verify your model file is valid
3. Ensure all dependencies are properly installed
4. Check that input images match your training data format