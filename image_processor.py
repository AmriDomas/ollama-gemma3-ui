from PIL import Image, ImageFilter, ImageEnhance
import pytesseract
import cv2
import numpy as np
from colorthief import ColorThief
import io
import tempfile

class ImageAnalyzer:
    def __init__(self):
        pass
    
    def extract_text(self, image):
        """Extract text from image using OCR"""
        try:
            # Convert PIL Image to bytes for pytesseract
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Use pytesseract
            text = pytesseract.image_to_string(Image.open(img_bytes))
            return text if text.strip() else "No text detected"
        except Exception as e:
            return f"OCR Error: {str(e)}"
    
    def get_image_info(self, image):
        """Get basic image information"""
        return {
            'size': image.size,
            'format': image.format,
            'mode': image.mode,
            'size_kb': len(image.tobytes()) / 1024
        }
    
    def analyze_colors(self, image):
        """Analyze dominant colors in image"""
        try:
            # Save image temporarily
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                image.save(tmp.name)
                
                # Use ColorThief
                color_thief = ColorThief(tmp.name)
                palette = color_thief.get_palette(color_count=5)
                
                # Convert RGB to hex
                colors = []
                for color in palette:
                    hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
                    colors.append((hex_color, 20))  # Mock percentage
                
                return colors
        except:
            return [("#FF0000", 100)]  # Fallback
    
    def detect_objects(self, image):
        """Simple object detection (placeholder)"""
        return ["Object detection requires YOLO/OpenCV setup"]
    
    def resize_image(self, image, width, height):
        """Resize image"""
        return image.resize((width, height), Image.Resampling.LANCZOS)
    
    def apply_filter(self, image, filter_type):
        """Apply filter to image"""
        filters = {
            'Grayscale': lambda img: img.convert('L'),
            'Blur': lambda img: img.filter(ImageFilter.BLUR),
            'Sharpen': lambda img: img.filter(ImageFilter.SHARPEN),
            'Edge Enhance': lambda img: img.filter(ImageFilter.EDGE_ENHANCE)
        }
        
        if filter_type in filters:
            return filters[filter_type](image)
        return image