import pytest
from pathlib import Path
from PIL import Image
import numpy as np
import os

# Import the engine from the main file
from Image_upscaler import UpscalerEngine

@pytest.fixture
def dummy_image(tmp_path):
    """Create a small dummy image for testing."""
    img_path = tmp_path / "test_input.png"
    # Create a 10x10 black image
    img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    img.save(img_path)
    return str(img_path)

@pytest.fixture
def engine():
    """Initialize the UpscalerEngine."""
    return UpscalerEngine()

def test_engine_initialization(engine):
    """Test if the engine initializes correctly."""
    assert hasattr(engine, 'upscale')
    # Check if AI flag is a boolean (depending on environment)
    assert isinstance(engine.has_ai, bool)

def test_upscale_logic_pil(engine, dummy_image):
    """Test if upscaling (via PIL fallback) works and produces correct dimensions."""
    scale = 2
    upscaled_img = engine.upscale(dummy_image, "Lanczos (Standard)", scale)
    
    assert isinstance(upscaled_img, Image.Image)
    assert upscaled_img.width == 20
    assert upscaled_img.height == 20

def test_upscale_invalid_path(engine):
    """Test that the engine raises ValueError for non-existent files."""
    with pytest.raises(ValueError, match="Input file not found."):
        engine.upscale("non_existent_file.jpg", "Lanczos (Standard)", 2)

def test_upscale_dimensions_all_scales(engine, dummy_image):
    """Verify different scale factors."""
    for scale in [2, 3, 4]:
        upscaled_img = engine.upscale(dummy_image, "Lanczos (Standard)", scale)
        assert upscaled_img.width == 10 * scale
        assert upscaled_img.height == 10 * scale

if __name__ == "__main__":
    pytest.main([__file__])
