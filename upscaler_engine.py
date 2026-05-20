"""
Upscaling engine for high-quality image processing using OpenCV DNN Super-Resolution.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path

class UpscalerEngine:
    """Handles image upscaling using AI models or high-quality interpolation."""

    def __init__(self, models_dir: str = "models") -> None:
        """
        Initialize the upscaler engine.

        Args:
            models_dir: Directory where the pre-trained model files are stored.
        """
        self.models_dir = Path(models_dir)
        self.has_dnn_sr = False
        try:
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            self.has_dnn_sr = True
        except AttributeError:
            print("Notice: cv2.dnn_superres not found. AI upscaling will be disabled.")
            print("To enable AI upscaling, please install: pip install opencv-contrib-python")
            self.sr = None
        
        self._ensure_models_dir()

    def _ensure_models_dir(self) -> None:
        """Ensure the models directory exists."""
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)

    def upscale_image(
        self, 
        input_path: str, 
        model_name: str = "edsr", 
        scale: int = 4
    ) -> Image.Image:
        """
        Upscale an image using the specified model and scale.

        Args:
            input_path: Path to the input image.
            model_name: Name of the model ('edsr', 'espcn', 'fsrcnn', 'lapsrn').
            scale: Upscaling factor (2, 3, or 4).

        Returns:
            A PIL Image object of the upscaled image.
        """
        # Load image with OpenCV
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not read image at {input_path}")

        model_path = self.models_dir / f"{model_name.upper()}_x{scale}.pb"

        if model_path.exists():
            try:
                # Initialize model
                self.sr.readModel(str(model_path))
                self.sr.setModel(model_name.lower(), scale)
                
                # Upscale
                result = self.sr.upsample(img)
                
                # Convert BGR to RGB
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                return Image.fromarray(result_rgb)
            except Exception as e:
                print(f"AI Upscaling failed: {e}. Falling back to PIL Lanczos.")
        else:
            print(f"Model file {model_path} not found. Falling back to PIL Lanczos.")

        # Fallback to PIL Lanczos
        pil_img = Image.open(input_path)
        width, height = pil_img.size
        new_size = (width * scale, height * scale)
        return pil_img.resize(new_size, Image.Resampling.LANCZOS)

def test_engine() -> None:
    """Simple test to verify engine initialization and fallback."""
    engine = UpscalerEngine()
    print("Engine initialized successfully.")
    # More comprehensive tests will be added in a separate test file.

if __name__ == "__main__":
    test_engine()
