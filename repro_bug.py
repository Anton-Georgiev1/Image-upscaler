from PIL import Image
import os
import torch
from unittest.mock import MagicMock, patch
from Image_upscaler import UpscalerEngine
from pathlib import Path

def test_reproduction():
    engine = UpscalerEngine()
    
    # Input size: 1024x1536
    w, h = 1024, 1536
    img = Image.new("RGB", (w, h))
    img_path = "repro.png"
    img.save(img_path)
    
    # Mock AI to return a 4x upscaled image (4096x6144)
    mock_out = Image.new("RGB", (w*4, h*4))
    
    with patch.object(UpscalerEngine, '_upscale_ai', return_value=mock_out):
        # User picked "4x"
        result = engine.upscale(img_path, "RealESRGAN (AI HD)", "4x")
        print(f"Result size with '4x': {result.size}")
        
        # User picked "4K (Ultra HD)"
        result_4k = engine.upscale(img_path, "RealESRGAN (AI HD)", "4K (Ultra HD)")
        print(f"Result size with '4K (Ultra HD)': {result_4k.size}")

    os.remove(img_path)

if __name__ == "__main__":
    test_reproduction()
