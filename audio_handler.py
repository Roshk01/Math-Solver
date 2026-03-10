# audio_handler.py
# Handles audio input - converts speech to text using Whisper
# Whisper = OpenAI's free speech recognition model

import os
import tempfile


def transcribe_audio(audio_file) -> dict:
    """
    Takes an uploaded audio file and converts speech to text.
    
    Returns:
        {
            "text": transcribed text,
            "confidence": float,
            "low_confidence": bool,
            "needs_confirmation": bool
        }
    """
    
    try:
        return _transcribe_with_whisper(audio_file)
    except ImportError:
        return {
            "text": "",
            "confidence": 0.0,
            "low_confidence": True,
            "needs_confirmation": True,
            "error": "Whisper not installed. Run: pip install openai-whisper"
        }
    except Exception as e:
        return {
            "text": "",
            "confidence": 0.0,
            "low_confidence": True,
            "needs_confirmation": True,
            "error": f"Audio transcription failed: {str(e)}"
        }


def _transcribe_with_whisper(audio_file) -> dict:
    """
    Transcribe audio using OpenAI Whisper (runs locally, FREE!)
    Whisper is excellent at understanding math phrases like:
    "square root of", "x raised to the power 2", etc.
    """
    import whisper
    
    # Load the small model (faster, good enough)
    # Options: tiny, base, small, medium, large
    model = whisper.load_model("base")
    
    # Save the uploaded file to a temp location
    with tempfile.NamedTemporaryFile(
        delete=False, 
        suffix=".wav"
    ) as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Transcribe
        result = model.transcribe(tmp_path, language="en")
        text = result["text"].strip()
        
        # Whisper doesn't give per-word confidence, 
        # but we can use segment-level probabilities
        segments = result.get("segments", [])
        if segments:
            avg_prob = sum(
                seg.get("avg_logprob", -1) 
                for seg in segments
            ) / len(segments)
            # Convert log prob to 0-1 scale
            confidence = max(0, min(1, (avg_prob + 2) / 2))
        else:
            confidence = 0.8  # Assume decent if no segment info
        
        return {
            "text": text,
            "confidence": round(confidence, 2),
            "low_confidence": confidence < 0.6,
            "needs_confirmation": confidence < 0.7,
            "method": "Whisper"
        }
    
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


def fix_math_speech(text: str) -> str:
    """
    Fix common speech-to-text issues with math phrases.
    
    Example:
    "square root of 16" → "sqrt(16)"
    "x to the power 2" → "x^2"
    "x squared" → "x^2"
    """
    
    import re
    
    replacements = [
        # Power expressions
        (r"(\w+) squared", r"\1^2"),
        (r"(\w+) cubed", r"\1^3"),
        (r"(\w+) to the power of (\w+)", r"\1^\2"),
        (r"(\w+) raised to (\w+)", r"\1^\2"),
        (r"(\w+) to the (\w+) power", r"\1^\2"),
        
        # Square root
        (r"square root of (\w+)", r"sqrt(\1)"),
        (r"root of (\w+)", r"sqrt(\1)"),
        
        # Fractions
        (r"(\w+) over (\w+)", r"\1/\2"),
        (r"(\w+) divided by (\w+)", r"\1/\2"),
        
        # Operations
        (r"plus", "+"),
        (r"minus", "-"),
        (r"times", "*"),
        (r"multiplied by", "*"),
        
        # Math constants
        (r"pi", "π"),
        (r"infinity", "∞"),
    ]
    
    result = text.lower()
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result
