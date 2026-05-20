import pytest
from pathlib import Path
from PIL import Image
import numpy as np
from unittest.mock import MagicMock, patch
import torch

# Import the engine from the main file
from Image_upscaler import UpscalerEngine

@pytest.fixture
def dummy_image(tmp_path: Path) -> str:
    """Create a small dummy image for testing."""
    img_path = tmp_path / "test_input.png"
    # Create a 10x10 black image
    img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    img.save(img_path)
    return str(img_path)

@pytest.fixture
def engine() -> UpscalerEngine:
    """Initialize the UpscalerEngine."""
    return UpscalerEngine()

def test_engine_initialization(engine: UpscalerEngine) -> None:
    """Test if the engine initializes correctly."""
    assert hasattr(engine, "upscale")
    assert isinstance(engine.has_ai, bool)

def test_upscale_logic_pil(engine: UpscalerEngine, dummy_image: str) -> None:
    """Test if upscaling (via PIL fallback) works and produces correct dimensions."""
    scale_val = "2x"
    model_type = "Lanczos (Fast CPU)"
    upscaled_img = engine.upscale(dummy_image, model_type, scale_val, perf_mode="Ultra (Max Speed)")
    
    assert isinstance(upscaled_img, Image.Image)
    assert upscaled_img.width == 20
    assert upscaled_img.height == 20

def test_upscale_invalid_path(engine: UpscalerEngine) -> None:
    """Test that the engine raises ValueError for non-existent files."""
    with pytest.raises(ValueError, match="Input file not found."):
        engine.upscale("non_existent_file.jpg", "Lanczos (Fast CPU)", "2x", perf_mode="Ultra (Max Speed)")

def test_upscale_dimensions_all_scales(engine: UpscalerEngine, dummy_image: str) -> None:
    """Verify different scale factors for PIL upscaling."""
    scales = {"2x": 2, "3x": 3, "4x": 4}
    for scale_str, scale_int in scales.items():
        upscaled_img = engine.upscale(dummy_image, "Lanczos (Fast CPU)", scale_str, perf_mode="Ultra (Max Speed)")
        assert upscaled_img.width == 10 * scale_int
        assert upscaled_img.height == 10 * scale_int

def test_upscale_4k_logic(engine: UpscalerEngine, dummy_image: str) -> None:
    """Verify 4K targeting logic."""
    # 4K is 3840x2160. Our dummy is 10x10.
    # The logic should upscale it to fit within 3840x2160 while maintaining aspect ratio.
    # Since 10x10 is square, it should become 2160x2160.
    upscaled_img = engine.upscale(dummy_image, "Lanczos (Fast CPU)", "4K (Ultra HD)", perf_mode="Ultra (Max Speed)")
    assert upscaled_img.width == 2160
    assert upscaled_img.height == 2160

@patch("Image_upscaler.ModelLoader")
@patch("Image_upscaler.ToTensor")
@patch("Image_upscaler.ToPILImage")
@patch("os.path.exists")
def test_upscale_ai_path(
    mock_exists: MagicMock,
    mock_to_pil: MagicMock,
    mock_to_tensor: MagicMock,
    mock_loader: MagicMock,
    engine: UpscalerEngine,
    dummy_image: str
) -> None:
    """Test the AI upscaling path using mocks."""
    # Setup mocks
    mock_exists.return_value = True
    
    mock_model = MagicMock()
    mock_model.scale = 4
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model
    # Mock model call: it takes a tensor and returns a scaled-up tensor
    def mock_model_call(x: torch.Tensor) -> torch.Tensor:
        # Simulate 4x upscale
        return torch.nn.functional.interpolate(x, scale_factor=4, mode='nearest')
    
    mock_model.side_effect = mock_model_call    
    mock_loader.return_value.load_from_file.return_value = mock_model
    
    # Mock ToTensor to return a dummy tensor with shape (3, 10, 10)
    # The implementation calls unsqueeze(0) which will make it (1, 3, 10, 10)
    mock_tensor = torch.zeros((3, 10, 10))
    mock_to_tensor.return_value.return_value = mock_tensor
    
    # Mock ToPILImage to return a dummy PIL image
    mock_result_img = Image.new("RGB", (40, 40))
    mock_to_pil.return_value.return_value = mock_result_img
    
    # Execute
    result = engine.upscale(dummy_image, "RealESRGAN (AI HD)", "4x", perf_mode="Ultra (Max Speed)")
    
    # Assertions
    assert result == mock_result_img
    mock_loader.return_value.load_from_file.assert_called_once()
    mock_to_tensor.return_value.assert_called()

def test_upscale_progress_callback(engine: UpscalerEngine, dummy_image: str) -> None:
    """Test if progress callback is called during upscaling."""
    progress_calls = []
    def callback(pct):
        progress_calls.append(pct)
        
    engine.upscale(dummy_image, "Lanczos (Fast CPU)", "2x", progress_callback=callback, perf_mode="Ultra (Max Speed)")
    
    assert len(progress_calls) > 0
    assert progress_calls[-1] == 1.0

def test_upscale_preserves_alpha(engine: UpscalerEngine, tmp_path: Path) -> None:
    """Verify that RGBA images keep their alpha channel after upscaling."""
    img_path = tmp_path / "test_alpha.png"
    # Create a 10x10 transparent image
    img = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
    img.save(img_path)
    
    # Test with PIL path
    upscaled_img = engine.upscale(str(img_path), "Lanczos (Fast CPU)", "2x", perf_mode="Ultra (Max Speed)")
    
    assert upscaled_img.mode == "RGBA"
    # Check if alpha channel still exists and has correct value at a pixel
    assert upscaled_img.getpixel((0, 0))[3] == 128

if __name__ == "__main__":
    pytest.main([__file__])
