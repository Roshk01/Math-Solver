# ocr_handler.py
# Handles image input - extracts text from photos/screenshots using OCR
# OCR = Optical Character Recognition (reading text from images)

import numpy as np
from PIL import Image
import io


def extract_text_from_image(image_file) -> dict:
    """
    Takes an uploaded image file and extracts text from it.
    
    Returns:
        {
            "text": extracted text,
            "confidence": float (0 to 1),
            "low_confidence": bool (True if AI should ask user to verify)
        }
    """
    
    # Try EasyOCR first (best for math/handwritten text)
    try:
        return _extract_with_easyocr(image_file)
    except ImportError:
        pass
    
    # Fallback: Try pytesseract
    try:
        return _extract_with_tesseract(image_file)
    except ImportError:
        pass
    
    # Final fallback: ask user to type manually
    return {
        "text": "",
        "confidence": 0.0,
        "low_confidence": True,
        "error": "OCR library not available. Please type the problem manually."
    }


def _extract_with_easyocr(image_file) -> dict:
    """Extract text using EasyOCR"""
    import easyocr
    
    # Initialize reader (English only for math)
    reader = easyocr.Reader(['en'], gpu=False)
    
    # Convert uploaded file to numpy array
    image = Image.open(image_file)
    img_array = np.array(image)
    
    # Run OCR
    results = reader.readtext(img_array)
    
    if not results:
        return {
            "text": "",
            "confidence": 0.0,
            "low_confidence": True,
            "error": "No text found in image"
        }
    
    # Combine all detected text
    texts = []
    confidences = []
    
    for (bbox, text, confidence) in results:
        texts.append(text)
        confidences.append(confidence)
    
    combined_text = " ".join(texts)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    return {
        "text": combined_text,
        "confidence": round(avg_confidence, 2),
        "low_confidence": avg_confidence < 0.6,  # Ask user if < 60% confident
        "method": "EasyOCR"
    }


def _extract_with_tesseract(image_file) -> dict:
    """Extract text using Tesseract (fallback)"""
    import pytesseract
    from pytesseract import Output
    
    image = Image.open(image_file)
    
    # Get text with confidence scores
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    
    texts = []
    confidences = []
    
    for i, text in enumerate(data["text"]):
        conf = int(data["conf"][i])
        if conf > 0 and text.strip():
            texts.append(text)
            confidences.append(conf / 100.0)
    
    if not texts:
        return {
            "text": "",
            "confidence": 0.0,
            "low_confidence": True,
            "error": "No text found in image"
        }
    
    combined_text = " ".join(texts)
    avg_confidence = sum(confidences) / len(confidences)
    
    return {
        "text": combined_text,
        "confidence": round(avg_confidence, 2),
        "low_confidence": avg_confidence < 0.6,
        "method": "Tesseract"
    }


def preprocess_math_text(text: str) -> str:
    """
    Fix common OCR mistakes in math problems.
    Example: "0" confused with "O", "1" confused with "l"
    """
    # Common OCR fixes for math
    fixes = {
        " O ": " 0 ",       # Letter O → number 0 (in math context)
        "×": "*",            # Multiplication sign
        "÷": "/",            # Division sign  
        "²": "^2",           # Superscript 2
        "³": "^3",           # Superscript 3
        "√": "sqrt",         # Square root symbol
        "≤": "<=",           # Less than or equal
        "≥": ">=",           # Greater than or equal
        "≠": "!=",           # Not equal
        "π": "pi",           # Pi
    }
    
    for wrong, right in fixes.items():
        text = text.replace(wrong, right)
    
    return text.strip()
