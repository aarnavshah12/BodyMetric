
# Body Measurement Analyzer

This application uses a Roboflow keypoint detection model to analyze body measurements from images, with a modern dark-themed GUI powered by ttkbootstrap.

## Features

- Upload images for analysis
- Input real-world eye distance for scale calibration
- Detect keypoints using Roboflow model
- Calculate body measurements (height, arm span, waist width, torso length, limb segments, etc.)
- Estimate hand length and head height using anatomical proportions
- Display original and processed images with keypoints
- Show detailed measurement results in a sortable table
- Modern dark UI with collapsible panels

## Setup

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project directory with your Roboflow credentials:
   ```
   ROBOFLOW_API_KEY="your_api_key_here"
   ROBOFLOW_WORKSPACE="your_workspace_name"
   ROBOFLOW_WORKFLOW_ID="your_workflow_id"
   ```

3. Run the application:
   ```
   python main.py
   ```

## Usage

1. **Upload Image**: Click "Select Image" to choose an image file.
2. **Enter Eye Distance**: Input the real-world distance between your eyes in centimeters (typically 6-7 cm for adults).
3. **Analyze**: Click "Analyze" to process the image and calculate measurements.
4. **View Results**: Check the results table and keypoints panel for calculated body dimensions.

## How It Works

1. The app uploads your image to the Roboflow model for keypoint detection.
2. It uses the eye distance as a reference to calculate the pixel-to-centimeter ratio.
3. Based on the detected keypoints, it calculates various body measurements:
   - Height (includes estimated head height above the eyes)
   - Arm span (shoulder width + both arms)
   - Torso length (vertical from eye to hip)
   - Limb segments (upper arm, forearm, thigh, shin, etc.)
   - Hand length (estimated from forearm length)
4. Results are displayed in a sortable table and as a list of detected keypoints.
5. Both original and processed images are shown in the GUI.

## Requirements

- Python 3.8+
- Internet connection for Roboflow API
- Valid Roboflow API key, workspace, and workflow ID (in `.env`)

## Notes

- Ensure good lighting and clear visibility of body parts for better keypoint detection.
- The accuracy of measurements depends on the quality of keypoint detection.
- Eye distance measurement is crucial for accurate scaling (typically 6-7 cm for adults).
- Head and hand lengths are estimated using standard anatomical proportions.
