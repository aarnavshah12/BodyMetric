"""
Body Measurement Application with Blackish Theme and 3x Larger Processed Image
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import requests
import base64
from io import BytesIO
import json
import math
from typing import List, Dict, Tuple, Optional
from inference_sdk import InferenceHTTPClient
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.style import Style
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class KeypointMapper:
    """Maps keypoint class names to body parts"""
    
    KEYPOINT_MAPPING = {
        'new-point-0': 'Nose',
        'new-point-1': 'Right Eye',
        'new-point-2': 'Left Eye',
        'new-point-3': 'Right Ear',
        'new-point-4': 'Left Ear',
        'new-point-5': 'Right Shoulder',
        'new-point-6': 'Left Shoulder',
        'new-point-7': 'Right Elbow',
        'new-point-8': 'Left Elbow',
        'new-point-9': 'Right Wrist',
        'new-point-10': 'Left Wrist',
        'new-point-11': 'Right Hip',
        'new-point-12': 'Left Hip',
        'new-point-13': 'Right Knee',
        'new-point-14': 'Left Knee',
        'new-point-15': 'Right Ankle',
        'new-point-16': 'Left Ankle'
    }
    
    @classmethod
    def get_body_part(cls, class_name: str) -> str:
        """Get body part name from class name"""
        return cls.KEYPOINT_MAPPING.get(class_name, class_name)
    
    @classmethod
    def find_keypoint_by_part(cls, keypoints: List[Dict], body_part: str) -> Optional[Dict]:
        """Find keypoint by body part name"""
        for kp in keypoints:
            if cls.get_body_part(kp['class']) == body_part:
                return kp
        return None

class MeasurementCalculator:
    """Calculate various body measurements from keypoints"""
    
    def __init__(self, keypoints: List[Dict], scale_ratio: float):
        self.keypoints = keypoints
        self.scale_ratio = scale_ratio
        self.mapper = KeypointMapper()
    
    def calculate_distance(self, point1: Dict, point2: Dict) -> float:
        """Calculate Euclidean distance between two points in pixels"""
        return math.sqrt( 
            (point1['x'] - point2['x'])**2 + 
            (point1['y'] - point2['y'])**2
        )
    
    def pixels_to_cm(self, pixel_distance: float) -> float:
        """Convert pixel distance to centimeters"""
        return pixel_distance / self.scale_ratio
    
    def get_eye_distance(self) -> Optional[float]:
        """Calculate eye distance in pixels"""
        left_eye = self.mapper.find_keypoint_by_part(self.keypoints, 'Left Eye')
        right_eye = self.mapper.find_keypoint_by_part(self.keypoints, 'Right Eye')
        
        if left_eye and right_eye:
            return self.calculate_distance(left_eye, right_eye)
        return None
    
    def get_shoulder_width(self) -> Optional[float]:
        """Calculate shoulder width in cm"""
        left_shoulder = self.mapper.find_keypoint_by_part(self.keypoints, 'Left Shoulder')
        right_shoulder = self.mapper.find_keypoint_by_part(self.keypoints, 'Right Shoulder')
        
        if left_shoulder and right_shoulder:
            pixel_distance = self.calculate_distance(left_shoulder, right_shoulder)
            return self.pixels_to_cm(pixel_distance)
        return None
    
    def get_arm_span(self) -> Optional[float]:
        """Calculate arm span by summing arm segments and shoulder width"""
        left_shoulder = self.mapper.find_keypoint_by_part(self.keypoints, 'Left Shoulder')
        left_elbow = self.mapper.find_keypoint_by_part(self.keypoints, 'Left Elbow')
        left_wrist = self.mapper.find_keypoint_by_part(self.keypoints, 'Left Wrist')
        right_shoulder = self.mapper.find_keypoint_by_part(self.keypoints, 'Right Shoulder')
        right_elbow = self.mapper.find_keypoint_by_part(self.keypoints, 'Right Elbow')
        right_wrist = self.mapper.find_keypoint_by_part(self.keypoints, 'Right Wrist')

        left_upper_arm = 0
        if left_shoulder and left_elbow:
            left_upper_arm = self.calculate_distance(left_shoulder, left_elbow)
        left_forearm = 0
        if left_elbow and left_wrist:
            left_forearm = self.calculate_distance(left_elbow, left_wrist)
        right_upper_arm = 0
        if right_shoulder and right_elbow:
            right_upper_arm = self.calculate_distance(right_shoulder, right_elbow)
        right_forearm = 0
        if right_elbow and right_wrist:
            right_forearm = self.calculate_distance(right_elbow, right_wrist)
        shoulder_width = 0
        if left_shoulder and right_shoulder:
            shoulder_width = self.calculate_distance(left_shoulder, right_shoulder)
        
        total_pixel_distance = left_upper_arm + left_forearm + right_upper_arm + right_forearm + shoulder_width
        if total_pixel_distance > 0:
            return self.pixels_to_cm(total_pixel_distance)
        return None
    
    def get_head_top_to_eye_length(self) -> Optional[float]:
        """Estimate vertical distance from top of head to eyes"""
        left_eye = self.mapper.find_keypoint_by_part(self.keypoints, 'Left Eye')
        right_eye = self.mapper.find_keypoint_by_part(self.keypoints, 'Right Eye')
        nose = self.mapper.find_keypoint_by_part(self.keypoints, 'Nose')

        if (left_eye or right_eye) and nose:
            eye_y = (left_eye['y'] + right_eye['y']) / 2 if left_eye and right_eye else (left_eye or right_eye)['y']
            eye_to_nose_vertical_dist = abs(eye_y - nose['y'])
            estimated_dist = eye_to_nose_vertical_dist * 2.0
            return estimated_dist
        return None

    def get_height(self) -> Optional[float]:
        """Calculate approximate height in cm, including estimated head height"""
        left_eye = self.mapper.find_keypoint_by_part(self.keypoints, 'Left Eye')
        right_eye = self.mapper.find_keypoint_by_part(self.keypoints, 'Right Eye')
        left_ankle = self.mapper.find_keypoint_by_part(self.keypoints, 'Left Ankle')
        right_ankle = self.mapper.find_keypoint_by_part(self.keypoints, 'Right Ankle')
        
        if (left_eye or right_eye) and (left_ankle or right_ankle):
            eye_y = (left_eye['y'] + right_eye['y']) / 2 if left_eye and right_eye else (left_eye or right_eye)['y']
            ankle_y = max(left_ankle['y'], right_ankle['y']) if left_ankle and right_ankle else (left_ankle or right_ankle)['y']
            body_pixel_distance = abs(eye_y - ankle_y)
            head_top_pixel_distance = self.get_head_top_to_eye_length() or 0
            total_pixel_distance = body_pixel_distance + head_top_pixel_distance
            return self.pixels_to_cm(total_pixel_distance)
        return None
    
    def get_waist_width(self) -> Optional[float]:
        """Calculate waist width in cm"""
        left_hip = self.mapper.find_keypoint_by_part(self.keypoints, 'Left Hip')
        right_hip = self.mapper.find_keypoint_by_part(self.keypoints, 'Right Hip')
        
        if left_hip and right_hip:
            pixel_distance = self.calculate_distance(left_hip, right_hip)
            return self.pixels_to_cm(pixel_distance)
        return None
    
    def get_arm_length(self, side: str = 'Left') -> Optional[float]:
        """Calculate arm length (shoulder to wrist) in cm"""
        shoulder = self.mapper.find_keypoint_by_part(self.keypoints, f'{side} Shoulder')
        wrist = self.mapper.find_keypoint_by_part(self.keypoints, f'{side} Wrist')
        
        if shoulder and wrist:
            pixel_distance = self.calculate_distance(shoulder, wrist)
            return self.pixels_to_cm(pixel_distance)
        return None
    
    def get_forearm_length(self, side: str = 'Left') -> Optional[float]:
        """Calculate forearm length (elbow to wrist) in cm"""
        elbow = self.mapper.find_keypoint_by_part(self.keypoints, f'{side} Elbow')
        wrist = self.mapper.find_keypoint_by_part(self.keypoints, f'{side} Wrist')
        
        if elbow and wrist:
            pixel_distance = self.calculate_distance(elbow, wrist)
            return self.pixels_to_cm(pixel_distance)
        return None
    
    def get_upper_arm_length(self, side: str = 'Left') -> Optional[float]:
        """Calculate upper arm length (shoulder to elbow) in cm"""
        shoulder = self.mapper.find_keypoint_by_part(self.keypoints, f'{side} Shoulder')
        elbow = self.mapper.find_keypoint_by_part(self.keypoints, f'{side} Elbow')
        
        if shoulder and elbow:
            pixel_distance = self.calculate_distance(shoulder, elbow)
            return self.pixels_to_cm(pixel_distance)
        return None
    
    def get_leg_length(self, side: str = 'Left') -> Optional[float]:
        """Calculate leg length (hip to ankle) in cm"""
        hip = self.mapper.find_keypoint_by_part(self.keypoints, f'{side} Hip')
        ankle = self.mapper.find_keypoint_by_part(self.keypoints, f'{side} Ankle')
        
        if hip and ankle:
            pixel_distance = self.calculate_distance(hip, ankle)
            return self.pixels_to_cm(pixel_distance)
        return None
    
    def get_thigh_length(self, side: str = 'Left') -> Optional[float]:
        """Calculate thigh length (hip to knee) in cm"""
        hip = self.mapper.find_keypoint_by_part(self.keypoints, f'{side} Hip')
        knee = self.mapper.find_keypoint_by_part(self.keypoints, f'{side} Knee')
        
        if hip and knee:
            pixel_distance = self.calculate_distance(hip, knee)
            return self.pixels_to_cm(pixel_distance)
        return None
    
    def get_shin_length(self, side: str = 'Left') -> Optional[float]:
        """Calculate shin length (knee to ankle) in cm"""
        knee = self.mapper.find_keypoint_by_part(self.keypoints, f'{side} Knee')
        ankle = self.mapper.find_keypoint_by_part(self.keypoints, f'{side} Ankle')
        
        if knee and ankle:
            pixel_distance = self.calculate_distance(knee, ankle)
            return self.pixels_to_cm(pixel_distance)
        return None
    
    def get_torso_length(self) -> Optional[float]:
        """Calculate torso length (eye to hip, vertical) in cm"""
        left_eye = self.mapper.find_keypoint_by_part(self.keypoints, 'Left Eye')
        right_eye = self.mapper.find_keypoint_by_part(self.keypoints, 'Right Eye')
        left_hip = self.mapper.find_keypoint_by_part(self.keypoints, 'Left Hip')
        right_hip = self.mapper.find_keypoint_by_part(self.keypoints, 'Right Hip')
        
        if (left_eye or right_eye) and (left_hip or right_hip):
            eye = left_eye if left_eye else right_eye
            hip = left_hip if left_hip else right_hip
            pixel_distance = abs(eye['y'] - hip['y'])
            return self.pixels_to_cm(pixel_distance)
        return None
    
    def get_all_measurements(self) -> Dict[str, float]:
        """Calculate all available measurements"""
        measurements = {}
        
        measurements['Shoulder Width'] = self.get_shoulder_width()
        measurements['Arm Span'] = self.get_arm_span()
        measurements['Height'] = self.get_height()
        measurements['Waist Width'] = self.get_waist_width()
        measurements['Torso Length'] = self.get_torso_length()
        
        measurements['Left Arm Length'] = self.get_arm_length('Left')
        measurements['Right Arm Length'] = self.get_arm_length('Right')
        measurements['Left Upper Arm'] = self.get_upper_arm_length('Left')
        measurements['Right Upper Arm'] = self.get_upper_arm_length('Right')
        measurements['Left Forearm'] = self.get_forearm_length('Left')
        measurements['Right Forearm'] = self.get_forearm_length('Right')
        
        measurements['Left Leg Length'] = self.get_leg_length('Left')
        measurements['Right Leg Length'] = self.get_leg_length('Right')
        measurements['Left Thigh'] = self.get_thigh_length('Left')
        measurements['Right Thigh'] = self.get_thigh_length('Right')
        measurements['Left Shin'] = self.get_shin_length('Left')
        measurements['Right Shin'] = self.get_shin_length('Right')
        
        return {k: v for k, v in measurements.items() if v is not None}

def format_measurement_results(measurements: Dict[str, float]) -> List[Dict[str, str]]:
    """Format measurement results as a list of dictionaries for table display"""
    categories = {
        'Overall': ['Height', 'Arm Span', 'Shoulder Width', 'Waist Width', 'Torso Length'],
        'Arms': ['Left Arm Length', 'Right Arm Length', 'Left Upper Arm', 'Right Upper Arm', 'Left Forearm', 'Right Forearm'],
        'Legs': ['Left Leg Length', 'Right Leg Length', 'Left Thigh', 'Right Thigh', 'Left Shin', 'Right Shin']
    }
    
    result = []
    for category, keys in categories.items():
        for key in keys:
            if key in measurements:
                result.append({
                    'Category': category,
                    'Measurement': key,
                    'Value': f"{measurements[key]:.1f} cm"
                })
    return result

class BodyMeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Body Measurement Analyzer")
        self.root.geometry("1400x800")
        
        # Initialize variables
        self.original_image = None
        self.processed_image = None
        self.eye_distance_pixels = None
        self.eye_distance_real = None
        self.keypoints = []
        self.scale_ratio = None
        self.results_expanded = True
        self.controls_expanded = True
        self.image_panel_size = (600, 500)  # Initial size, updated dynamically
        
        # Initialize Roboflow client
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=os.getenv("ROBOFLOW_API_KEY")
        )
        
        # Configure style
        self.style = Style(theme='superhero')
        self.style.configure('TFrame', background='#1a1a1a')  # Deep black background
        self.style.configure('Card.TFrame', background='#1a1a1a', relief='flat', bordercolor='#2c2c2c', borderwidth=1)
        self.style.configure('TLabel', background='#1a1a1a', foreground='#ffffff')
        self.style.configure('primary.TLabel', foreground='#0d6efd', font=('Segoe UI', 14, 'bold'))
        self.style.configure('secondary.TLabel', foreground='#6c757d')
        self.style.configure('primary.TButton', background='#0d6efd', foreground='#ffffff', font=('Segoe UI', 10))
        self.style.configure('success.TButton', background='#28a745', foreground='#ffffff', font=('Segoe UI', 10))
        self.style.configure('primary.Outline.TButton', background='#1a1a1a', foreground='#0d6efd', font=('Segoe UI', 10))
        self.style.configure('info.TEntry', fieldbackground='#2c2c2c', foreground='#ffffff')
        self.style.configure('primary.Treeview', background='#2c2c2c', fieldbackground='#2c2c2c', foreground='#ffffff')
        self.style.configure('primary.Treeview.Heading', background='#0d6efd', foreground='#ffffff', font=('Segoe UI', 10, 'bold'))
        self.style.configure('info.TLabel', background='#1a1a1a', foreground='#17a2b8')
        self.style.configure('info.Horizontal.TProgressbar', troughcolor='#2c2c2c', background='#17a2b8')
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel for controls and results
        left_panel = ttk.Frame(main_frame, style='Card.TFrame')
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 20))
        
        # Controls section (collapsible)
        controls_header = ttk.Frame(left_panel)
        controls_header.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        ttk.Label(controls_header, text="Controls", style='primary.TLabel').grid(row=0, column=0, sticky=tk.W)
        ttk.Button(controls_header, text="▼", command=self.toggle_controls, style='primary.Outline.TButton', width=3).grid(row=0, column=1, sticky=tk.E)
        
        self.controls_frame = ttk.Frame(left_panel, padding=10)
        self.controls_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Image upload
        ttk.Label(self.controls_frame, text="Upload Image", font=('Segoe UI', 10)).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Button(self.controls_frame, text="Select Image", command=self.select_image, style='primary.TButton', width=15).grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Eye distance input
        ttk.Label(self.controls_frame, text="Eye Distance (cm)", font=('Segoe UI', 10)).grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        self.eye_distance_var = tk.StringVar()
        ttk.Entry(self.controls_frame, textvariable=self.eye_distance_var, width=10, style='info.TEntry').grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Analyze button
        self.analyze_button = ttk.Button(self.controls_frame, text="Analyze", command=self.process_image, style='success.TButton', width=15)
        self.analyze_button.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Results section (collapsible)
        results_header = ttk.Frame(left_panel)
        results_header.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 10))
        ttk.Label(results_header, text="Results", style='primary.TLabel').grid(row=0, column=0, sticky=tk.W)
        ttk.Button(results_header, text="▼", command=self.toggle_results, style='primary.Outline.TButton', width=3).grid(row=0, column=1, sticky=tk.E)
        
        self.results_frame = ttk.Frame(left_panel, padding=10)
        self.results_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Results table
        self.results_tree = ttk.Treeview(self.results_frame, columns=('Category', 'Measurement', 'Value'), show='headings', style='primary.Treeview')
        self.results_tree.heading('Category', text='Category')
        self.results_tree.heading('Measurement', text='Measurement')
        self.results_tree.heading('Value', text='Value (cm)')
        self.results_tree.column('Category', width=100)
        self.results_tree.column('Measurement', width=150)
        self.results_tree.column('Value', width=100)
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.results_tree.yview, style='primary.Vertical.TScrollbar')
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.results_frame.rowconfigure(0, weight=1)
        self.results_frame.columnconfigure(0, weight=1)
        
        # Keypoints text
        self.keypoints_text = tk.Text(self.results_frame, height=10, width=40, font=('Segoe UI', 9), wrap=tk.WORD, bg='#212529', fg='#ffffff', bd=0)
        self.keypoints_text.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        keypoints_scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.keypoints_text.yview, style='primary.Vertical.TScrollbar')
        self.keypoints_text.configure(yscrollcommand=keypoints_scrollbar.set)
        keypoints_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        
        # Right panel for images
        self.image_panel = ttk.Frame(main_frame, style='Card.TFrame')
        self.image_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.image_panel.rowconfigure(1, weight=1)  # Original image
        self.image_panel.rowconfigure(3, weight=3)  # Processed image (3x area)
        
        # Original image
        ttk.Label(self.image_panel, text="Original Image", font=('Segoe UI', 12, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        self.original_label = ttk.Label(self.image_panel, text="No image selected", style='secondary.TLabel', background='#1a1a1a', anchor='center')
        self.original_label.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        
        # Processed image
        ttk.Label(self.image_panel, text="Processed Image", font=('Segoe UI', 12, 'bold')).grid(row=2, column=0, sticky=tk.W, pady=(0, 10))
        self.processed_label = ttk.Label(self.image_panel, text="No processed image", style='secondary.TLabel', background='#1a1a1a', anchor='center')
        self.processed_label.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar with progress
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var, style='info.TLabel', padding=5).grid(row=0, column=0, sticky=tk.W)
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', style='info.Horizontal.TProgressbar')
        self.progress.grid(row=0, column=1, sticky=(tk.E), padx=10)
        status_frame.columnconfigure(0, weight=1)
        
        # Bind window resize to update image panel size
        self.image_panel.bind('<Configure>', self.update_image_panel_size)
    
    def update_image_panel_size(self, event=None):
        """Update the maximum image size based on the image panel's dimensions"""
        panel_width = self.image_panel.winfo_width()
        panel_height = self.image_panel.winfo_height()
        # Account for padding and labels
        total_height = panel_height - 60  # Subtract padding and label heights
        # Allocate 1/4 height to original, 3/4 to processed (3x area)
        original_height = total_height // 4
        processed_height = (total_height * 3) // 4
        self.image_panel_size = {
            'original': (max(100, panel_width - 40), max(100, original_height)),
            'processed': (max(100, panel_width - 40), max(100, processed_height))
        }
        # Update displayed images if they exist
        if self.original_image:
            self.update_original_image()
        if self.processed_image:
            self.update_processed_image()
    
    def toggle_controls(self):
        """Toggle visibility of controls frame"""
        self.controls_expanded = not self.controls_expanded
        if self.controls_expanded:
            self.controls_frame.grid()
            self.controls_header.winfo_children()[1].configure(text="▼")
        else:
            self.controls_frame.grid_remove()
            self.controls_header.winfo_children()[1].configure(text="▶")
    
    def toggle_results(self):
        """Toggle visibility of results frame"""
        self.results_expanded = not self.results_expanded
        if self.results_expanded:
            self.results_frame.grid()
            self.results_header.winfo_children()[1].configure(text="▼")
        else:
            self.results_frame.grid_remove()
            self.results_header.winfo_children()[1].configure(text="▶")
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        
        if file_path:
            try:
                self.original_image = Image.open(file_path)
                self.image_path = file_path
                self.update_original_image()
                self.status_var.set(f"Image loaded: {file_path.split('/')[-1]}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}", parent=self.root, icon='error')
    
    def update_original_image(self):
        """Update the original image display to fit the panel"""
        if self.original_image:
            display_img = self.resize_image_for_display(self.original_image, 'original')
            photo = ImageTk.PhotoImage(display_img)
            self.original_label.configure(image=photo, text="")
            self.original_label.image = photo
    
    def update_processed_image(self):
        """Update the processed image display to fit the panel"""
        if self.processed_image:
            display_img = self.resize_image_for_display(self.processed_image, 'processed')
            photo = ImageTk.PhotoImage(display_img)
            self.processed_label.configure(image=photo, text="")
            self.processed_label.image = photo
    
    def resize_image_for_display(self, image, image_type):
        """Resize image to fit within the image panel while maintaining aspect ratio"""
        img_width, img_height = image.size
        max_width, max_height = self.image_panel_size[image_type]
        scale = min(max_width / img_width, max_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def process_image(self):
        if not self.original_image:
            messagebox.showerror("Error", "Please select an image first", parent=self.root, icon='error')
            return
        
        if not self.eye_distance_var.get():
            messagebox.showerror("Error", "Please enter the eye distance", parent=self.root, icon='error')
            return
        
        try:
            self.eye_distance_real = float(self.eye_distance_var.get())
            if self.eye_distance_real <= 0:
                raise ValueError("Eye distance must be positive")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid positive number for eye distance", parent=self.root, icon='error')
            return
        
        self.status_var.set("Analyzing image...")
        self.progress.start(10)
        self.analyze_button.configure(state='disabled')
        self.root.update()
        
        try:
            result = self.client.run_workflow(
                workspace_name=os.getenv("ROBOFLOW_WORKSPACE"),
                workflow_id=os.getenv("ROBOFLOW_WORKFLOW_ID"),
                images={"image": self.image_path},
                use_cache=True
            )
            
            self.extract_keypoints(result)
            self.load_visualization(result)
            self.calculate_measurements()
            
            self.status_var.set("Analysis complete")
            
        except Exception as e:
            print(f"Debug: Error in process_image: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to process image: {str(e)}", parent=self.root, icon='error')
            self.status_var.set("Analysis failed")
        
        self.progress.stop()
        self.analyze_button.configure(state='normal')
    
    def extract_keypoints(self, result):
        """Extract keypoints from Roboflow result"""
        self.keypoints = []
        try:
            if isinstance(result, list) and result:
                outer_predictions = result[0].get('predictions', {})
                person_detections = outer_predictions.get('predictions', [])
                
                if person_detections and isinstance(person_detections, list):
                    keypoints_list = person_detections[0].get('keypoints', [])
                    for keypoint in keypoints_list:
                        if all(key in keypoint for key in ['class_id', 'class', 'x', 'y']):
                            self.keypoints.append({
                                'class_id': keypoint.get('class_id'),
                                'class': keypoint.get('class'),
                                'x': keypoint.get('x'),
                                'y': keypoint.get('y'),
                                'confidence': keypoint.get('confidence')
                            })
            
            self.keypoints.sort(key=lambda x: x['class_id'] if x.get('class_id') is not None else 999)
            
        except Exception as e:
            print(f"Error extracting keypoints: {e}")
            import traceback
            traceback.print_exc()
    
    def load_visualization(self, result):
        """Load and display the visualization image"""
        try:
            vis_img = None
            if isinstance(result, list) and result and isinstance(result[0], dict):
                vis_data = result[0].get("keypoint_visualization")
                
                if vis_data:
                    if isinstance(vis_data, str) and vis_data.startswith("data:image"):
                        base64_string = vis_data.split(",")[1] if "," in vis_data else vis_data
                        img_data = base64.b64decode(base64_string)
                        vis_img = Image.open(BytesIO(img_data))
                    elif isinstance(vis_data, str) and (vis_data.startswith("/9j/") or vis_data.startswith("iVBOR")):
                        img_data = base64.b64decode(vis_data)
                        vis_img = Image.open(BytesIO(img_data))
                    elif isinstance(vis_data, str) and vis_data.startswith("http"):
                        response = requests.get(vis_data, timeout=10)
                        response.raise_for_status()
                        vis_img = Image.open(BytesIO(response.content))
            
            if vis_img:
                self.processed_image = vis_img
                self.update_processed_image()
                vis_img.save("visualization_output.png")
            else:
                self.processed_label.configure(image=None, text="No processed image available")
                self.processed_label.image = None
                
        except Exception as e:
            print(f"Error loading visualization: {e}")
            self.processed_label.configure(image=None, text="No processed image available")
            self.processed_label.image = None
    
    def calculate_measurements(self):
        """Calculate body measurements based on keypoints and eye distance"""
        if len(self.keypoints) < 2:
            self.results_tree.delete(*self.results_tree.get_children())
            self.keypoints_text.delete(1.0, tk.END)
            self.keypoints_text.insert(tk.END, "Insufficient keypoints detected.\nPlease ensure a clear frontal view of a person.", 'error')
            self.keypoints_text.tag_configure('error', foreground='#dc3545')
            return
        
        calculator = MeasurementCalculator(self.keypoints, 1.0)
        self.eye_distance_pixels = calculator.get_eye_distance()
        
        if self.eye_distance_pixels:
            self.scale_ratio = self.eye_distance_pixels / self.eye_distance_real
            reference_part = "Eye Distance"
        else:
            left_shoulder = KeypointMapper.find_keypoint_by_part(self.keypoints, 'Left Shoulder')
            right_shoulder = KeypointMapper.find_keypoint_by_part(self.keypoints, 'Right Shoulder')
            if left_shoulder and right_shoulder:
                self.eye_distance_pixels = calculator.calculate_distance(left_shoulder, right_shoulder)
                self.scale_ratio = self.eye_distance_pixels / self.eye_distance_real
                reference_part = "Shoulder Width"
            else:
                self.results_tree.delete(*self.results_tree.get_children())
                self.keypoints_text.delete(1.0, tk.END)
                self.keypoints_text.insert(tk.END, "Unable to calculate scale: no eye or shoulder keypoints detected.\nPlease ensure a clear view of face or shoulders.", 'error')
                self.keypoints_text.tag_configure('error', foreground='#dc3545')
                return
        
        calculator.scale_ratio = self.scale_ratio
        measurements = calculator.get_all_measurements()
        self.display_results(measurements, reference_part)
    
    def display_results(self, measurements, reference_part):
        """Display measurement results in the table and keypoints in text"""
        self.results_tree.delete(*self.results_tree.get_children())
        self.keypoints_text.delete(1.0, tk.END)
        
        # Display keypoints
        self.keypoints_text.tag_configure('header', font=('Segoe UI', 10, 'bold'), foreground='#0d6efd')
        self.keypoints_text.tag_configure('item', font=('Segoe UI', 9))
        self.keypoints_text.insert(tk.END, f"Scale: {reference_part} ({self.eye_distance_real:.1f} cm, {self.eye_distance_pixels:.1f} pixels, {self.scale_ratio:.2f} pixels/cm)\n\n", 'header')
        self.keypoints_text.insert(tk.END, "Detected Keypoints:\n", 'header')
        mapper = KeypointMapper()
        for i, kp in enumerate(self.keypoints):
            body_part = mapper.get_body_part(kp['class'])
            self.keypoints_text.insert(tk.END, f"{i+1}. {body_part}: ({kp['x']:.0f}, {kp['y']:.0f}) [Conf: {kp['confidence']:.3f}]\n", 'item')
        
        # Display measurements in table
        formatted_results = format_measurement_results(measurements)
        for item in formatted_results:
            self.results_tree.insert('', tk.END, values=(item['Category'], item['Measurement'], item['Value']))
        
        if not measurements:
            self.keypoints_text.insert(tk.END, "\nNo measurements calculated.\nPlease ensure clear keypoints in the image.", 'error')
            self.keypoints_text.tag_configure('error', foreground='#dc3545')

def main():
    root = ttk.Window(themename='superhero')
    app = BodyMeasurementApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()