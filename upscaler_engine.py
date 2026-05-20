"""
Upscaling engine for high-quality image processing using OpenCV DNN Super-Resolution.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import sys

try:
    import torch
    from super_image import EdsrModel, ImageLoader
    HAS_SUPER_IMAGE = True
except ImportError:
    HAS_SUPER_IMAGE = False

class UpscalerEngine:
    """Handles image upscaling using AI models or high-quality interpolation."""

    def __init__(self, models_dir: str = "models") -> None:
        """
        Initialize the upscaler engine.

        Args:
            models_dir: Directory where the pre-trained model files are stored.
        """
        self.models_dir = Path(models_dir)
        self.has_super_image = HAS_SUPER_IMAGE
        self.has_torch = HAS_SUPER_IMAGE or 'torch' in sys.modules
        
        self.has_dnn_sr = False
        try:
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            self.has_dnn_sr = True
        except AttributeError:
            self.sr = None
        
        self._ensure_models_dir()

    def _ensure_models_dir(self) -> None:
        """Ensure the models directory exists."""
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)

    def upscale_image(
        self, 
        input_path: str, 
        model_name: str = "real-esrgan", 
        scale: int = 4
    ) -> Image.Image:
        """
        Upscale an image using the specified model and scale.
        Only supports Real-ESRGAN and EDSR (HD).
        """
        if not Path(input_path).exists():
            raise ValueError(f"Input image file does not exist: {input_path}")
            
        model_name = model_name.lower()
        
        # Priority 1: Real-ESRGAN (The "Best" HD model)
        if "real-esrgan" in model_name:
            return self._upscale_realesrgan(input_path, scale)
            
        # Priority 2: EDSR (HD)
        if self.has_super_image and "edsr" in model_name:
            return self._upscale_super_image(input_path, scale)
        
        # Priority 3: PIL Fallback (if AI fails or unsupported)
        return self._pil_fallback(input_path, scale)

    def _upscale_realesrgan(self, input_path: str, scale: int) -> Image.Image:
        """Upscale using the best available Real-ESRGAN implementation."""
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
            model_path = self.models_dir / f"RealESRGAN_x{scale}plus.pth"
            
            upsampler = RealESRGANer(
                scale=scale,
                model_path=str(model_path),
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
            print(f"Real-ESRGAN failed: {e}. Falling back to EDSR (HD).")
            return self._upscale_super_image(input_path, scale)

    def _upscale_super_image(self, input_path: str, scale: int) -> Image.Image:
        """Upscale using super-image (PyTorch)."""
        if not self.has_super_image:
            return self._pil_fallback(input_path, scale)
            
        try:
            image = Image.open(input_path)
            model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=scale)
            inputs = ImageLoader.load_image(image)
            preds = model(inputs)
            return self._tensor_to_pil(preds)
        except Exception as e:
            print(f"Super-image upscaling failed: {e}. Falling back.")
            return self._pil_fallback(input_path, scale)

    def _pil_fallback(self, input_path: str, scale: int) -> Image.Image:
        """High-quality PIL Lanczos fallback."""
        pil_img = Image.open(input_path)
        width, height = pil_img.size
        new_size = (width * scale, height * scale)
        return pil_img.resize(new_size, Image.Resampling.LANCZOS)

    def _tensor_to_pil(self, tensor) -> Image.Image:
        """Convert a torch tensor from super-image to a PIL Image."""
        from torchvision.transforms import ToPILImage
        tensor = tensor.cpu().detach().squeeze(0)
        return ToPILImage()(tensor)

def test_engine() -> None:
    """Simple test to verify engine initialization."""
    engine = UpscalerEngine()
    print("Engine initialized successfully.")

if __name__ == "__main__":
    test_engine()
