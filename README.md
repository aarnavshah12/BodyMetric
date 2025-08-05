# Body Measurement Analysis GUI

This application uses a Roboflow keypoint detection model to analyze body measurements from images.

## Features

- Upload images for analysis
- Input real-world eye distance for scale calibration
- Detect keypoints using Roboflow model
- Calculate body measurements (height, arm span, waist width, etc.)
- Display original and processed images with keypoints
- Show detailed measurement results

## Setup

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python body_measurement_gui.py
   ```

## Usage

1. **Upload Image**: Click "Select Image" to choose an image file
2. **Enter Eye Distance**: Input the real-world distance between your eyes in centimeters (typically 6-7 cm for adults)
3. **Process Image**: Click "Process Image" to analyze the image and calculate measurements
4. **View Results**: Check the measurements panel for calculated body dimensions

## How It Works

1. The app uploads your image to the Roboflow model for keypoint detection
2. It uses the eye distance as a reference to calculate the pixel-to-centimeter ratio
3. Based on the detected keypoints, it calculates various body measurements
4. Results are displayed with both the original and processed images

## Requirements

- Python 3.7+
- Internet connection for Roboflow API
- Valid Roboflow API key (configured in the code)

## Notes

- Ensure good lighting and clear visibility of body parts for better keypoint detection
- The accuracy of measurements depends on the quality of keypoint detection
- Eye distance measurement is crucial for accurate scaling - typically 6-7 cm for adults
