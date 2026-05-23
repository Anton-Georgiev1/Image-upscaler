import pytest
from pathlib import Path
from PIL import Image
import numpy as np
from unittest.mock import MagicMock, patch
import sys

sys.modules['customtkinter'] = MagicMock()
sys.modules['tkinterdnd2'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.transforms'] = MagicMock()
sys.modules['spandrel'] = MagicMock()

from Image_upscaler import UpscalerEngine

@pytest.fixture
def dummy_image(tmp_path: Path) -> str:
    img_path = tmp_path / "test_input.png"
    img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    img.save(img_path)
    return str(img_path)

@pytest.fixture
def engine() -> UpscalerEngine:
    return UpscalerEngine()

def test_engine_initialization(engine: UpscalerEngine) -> None:
    assert hasattr(engine, "upscale")
    assert isinstance(engine.has_ai, bool)

def test_upscale_logic_pil(engine: UpscalerEngine, dummy_image: str) -> None:
    upscaled_img = engine.upscale(dummy_image, "Lanczos (Fast CPU)", "2x", perf_mode="Ultra (Max Speed)")
    assert isinstance(upscaled_img, Image.Image)
    assert upscaled_img.width == 20
    assert upscaled_img.height == 20

def test_upscale_invalid_path(engine: UpscalerEngine) -> None:
    with pytest.raises(ValueError, match="Input file not found."):
        engine.upscale("non_existent_file.jpg", "Lanczos (Fast CPU)", "2x", perf_mode="Ultra (Max Speed)")

def test_upscale_dimensions_all_scales(engine: UpscalerEngine, dummy_image: str) -> None:
    scales = {"2x": 2, "3x": 3, "4x": 4}
    for scale_str, scale_int in scales.items():
        upscaled_img = engine.upscale(dummy_image, "Lanczos (Fast CPU)", scale_str, perf_mode="Ultra (Max Speed)")
        assert upscaled_img.width == 10 * scale_int
        assert upscaled_img.height == 10 * scale_int

def test_upscale_4k_logic(engine: UpscalerEngine, dummy_image: str) -> None:
    upscaled_img = engine.upscale(dummy_image, "Lanczos (Fast CPU)", "4K (Ultra HD)", perf_mode="Ultra (Max Speed)")
    assert upscaled_img.width == 2160
    assert upscaled_img.height == 2160

@patch("Image_upscaler.ModelLoader")
@patch("Image_upscaler.ToTensor")
@patch("Image_upscaler.ToPILImage")
@patch("os.path.exists")
def test_upscale_ai_path(mock_exists, mock_to_pil, mock_to_tensor, mock_loader, engine, dummy_image):
    mock_exists.return_value = True
    
    mock_model = MagicMock()
    mock_model.scale = 4
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model
    mock_loader.return_value.load_from_file.return_value = mock_model
    
    mock_result_img = Image.new("RGB", (40, 40))
    mock_to_pil.return_value.return_value = mock_result_img
    
    # Strictly scope this patch so it does not infect other test cases below
    with patch.object(engine, '_process_with_tiling', return_value=MagicMock()):
        result = engine.upscale(dummy_image, "RealESRGAN (AI HD)", "4x", perf_mode="Ultra (Max Speed)")
        mock_loader.return_value.load_from_file.assert_called_once()

def test_upscale_progress_callback(engine: UpscalerEngine, dummy_image: str) -> None:
    progress_calls = []
    def callback(pct):
        progress_calls.append(pct)
        
    engine.upscale(dummy_image, "Lanczos (Fast CPU)", "2x", progress_callback=callback, perf_mode="Ultra (Max Speed)")
    assert len(progress_calls) > 0
    assert progress_calls[-1] == 1.0

def test_upscale_preserves_alpha(engine: UpscalerEngine, tmp_path: Path) -> None:
    img_path = tmp_path / "test_alpha.png"
    img = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
    img.save(img_path)
    
    upscaled_img = engine.upscale(str(img_path), "Lanczos (Fast CPU)", "2x", perf_mode="Ultra (Max Speed)")
    assert upscaled_img.mode == "RGBA"
    assert upscaled_img.getpixel((0, 0))[3] == 128

def test_upscale_4x_portrait_no_cap(engine: UpscalerEngine, tmp_path: Path) -> None:
    img_path = tmp_path / "test_portrait.png"
    img = Image.new("RGB", (1024, 1536))
    img.save(img_path)
    
    result = engine.upscale(str(img_path), "Lanczos (Fast CPU)", "4x")
    assert result.width == 4096
    assert result.height == 6144
