"""
Upscaling engine for high-quality image processing using OpenCV DNN Super-Resolution.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pathlib import Path

try:
    import torch
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

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
            self.sr = None
        
        self.has_torch = HAS_TORCH
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
        """
        # Load image with OpenCV for DNN/Torch, PIL for fallback
        if model_name.lower() == "realesrgan":
            return self._upscale_realesrgan(input_path, scale)
            
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not read image at {input_path}")

        model_path = self.models_dir / f"{model_name.upper()}_x{scale}.pb"

        if self.has_dnn_sr and model_path.exists():
            try:
                self.sr.readModel(str(model_path))
                self.sr.setModel(model_name.lower(), scale)
                result = self.sr.upsample(img)
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                return Image.fromarray(result_rgb)
            except Exception as e:
                print(f"AI Upscaling failed: {e}. Falling back to PIL Lanczos.")
        
        # Fallback to PIL Lanczos
        pil_img = Image.open(input_path)
        width, height = pil_img.size
        new_size = (width * scale, height * scale)
        return pil_img.resize(new_size, Image.Resampling.LANCZOS)

    def _upscale_realesrgan(self, input_path: str, scale: int) -> Image.Image:
        """Upscale using Real-ESRGAN (requires torch and realesrgan)."""
        if not self.has_torch:
            print("Real-ESRGAN requires 'torch' and 'realesrgan' packages. Falling back.")
            return self.upscale_image(input_path, "pil", scale)

        try:
            # Note: This is a simplified integration. 
            # In a real scenario, we'd need the .pth model file.
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
            upsampler = RealESRGANer(
                scale=scale,
                model_path=str(self.models_dir / f"RealESRGAN_x{scale}.pth"),
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=True if torch.cuda.is_available() else False
            )
            
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            output, _ = upsampler.enhance(img, outscale=scale)
            
            if len(output.shape) == 3 and output.shape[2] == 4: # RGBA
                output_rgb = cv2.cvtColor(output, cv2.COLOR_BGRA2RGBA)
            else:
                output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                
            return Image.fromarray(output_rgb)
        except Exception as e:
            print(f"Real-ESRGAN failed: {e}. Falling back.")
            return self.upscale_image(input_path, "pil", scale)

def test_engine() -> None:
    """Simple test to verify engine initialization and fallback."""
    engine = UpscalerEngine()
    print("Engine initialized successfully.")
    # More comprehensive tests will be added in a separate test file.

if __name__ == "__main__":
    test_engine()
