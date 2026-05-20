import pytest
from pathlib import Path
from PIL import Image
import numpy as np
import os
from upscaler_engine import UpscalerEngine

@pytest.fixture
def dummy_image(tmp_path):
    """Create a small dummy image for testing."""
    img_path = tmp_path / "test_image.png"
    img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    img.save(img_path)
    return str(img_path)

@pytest.fixture
def engine():
    """Initialize the UpscalerEngine."""
    return UpscalerEngine(models_dir="test_models")

def test_engine_initialization(engine):
    """Test if engine initializes and creates models directory."""
    assert os.path.exists("test_models")
    # Should initialize regardless of whether dnn_sr is available
    assert hasattr(engine, 'has_dnn_sr')

def test_upscale_fallback_pil(engine, dummy_image):
    """Test if upscaling falls back to PIL when models are missing."""
    # Scale x2
    upscaled_img = engine.upscale_image(dummy_image, model_name="edsr", scale=2)
    
    assert isinstance(upscaled_img, Image.Image)
    assert upscaled_img.size == (20, 20)

def test_upscale_invalid_path(engine):
    """Test behavior with invalid image path."""
    with pytest.raises(ValueError):
        engine.upscale_image("non_existent_image.png")

if __name__ == "__main__":
    # Clean up test models dir if it exists from previous runs
    if os.path.exists("test_models"):
        import shutil
        shutil.rmtree("test_models")
    
    pytest.main([__file__])
