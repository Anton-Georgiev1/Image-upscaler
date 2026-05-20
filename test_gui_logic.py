import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import os
import sys

# Define dummy base classes to avoid metaclass conflict during testing
class DummyCTk:
    def __init__(self, *args, **kwargs): pass
    def grid_columnconfigure(self, *args, **kwargs): pass
    def grid_rowconfigure(self, *args, **kwargs): pass
    def title(self, *args, **kwargs): pass
    def geometry(self, *args, **kwargs): pass
    def after(self, *args, **kwargs): pass

class DummyDnDWrapper:
    def __init__(self, *args, **kwargs): pass
    def drop_target_register(self, *args, **kwargs): pass
    def dnd_bind(self, *args, **kwargs): pass

# Mock the modules before importing Image_upscaler
mock_ctk = MagicMock()
mock_ctk.CTk = DummyCTk
mock_ctk.CTkFrame = MagicMock()
mock_ctk.CTkLabel = MagicMock()
mock_ctk.CTkButton = MagicMock()
mock_ctk.CTkOptionMenu = MagicMock()
mock_ctk.CTkProgressBar = MagicMock()
mock_ctk.CTkFont = MagicMock()

mock_dnd = MagicMock()
mock_dnd.TkinterDnD.DnDWrapper = DummyDnDWrapper
mock_dnd.DND_ALL = "all"

mock_torch = MagicMock()
mock_torchvision = MagicMock()
mock_spandrel = MagicMock()

with patch.dict(sys.modules, {
    'customtkinter': mock_ctk, 
    'tkinterdnd2': mock_dnd,
    'torch': mock_torch,
    'torchvision': mock_torchvision,
    'torchvision.transforms': mock_torchvision.transforms,
    'spandrel': mock_spandrel
}):
    from Image_upscaler import Image_upscaler

@pytest.fixture
def app():
    with patch('Image_upscaler.UpscalerEngine'), \
         patch('Image_upscaler.TkinterDnD._require'):
        # We need to manually mock the widgets because they are created in __init__
        with patch('customtkinter.CTkFrame'), \
             patch('customtkinter.CTkLabel'), \
             patch('customtkinter.CTkButton'), \
             patch('customtkinter.CTkOptionMenu'), \
             patch('customtkinter.CTkProgressBar'):
            app = Image_upscaler()
            return app

def test_load_image(app):
    """Test that load_image sets the input path and updates UI state."""
    app.btn_run = MagicMock()
    app.btn_save = MagicMock()
    app.show_preview = MagicMock()
    
    test_path = "test.png"
    app.load_image(test_path)
    
    assert app.input_path == test_path
    app.btn_run.configure.assert_called_with(state="normal")
    app.btn_save.configure.assert_called_with(state="disabled")
    app.show_preview.assert_called_with(test_path)

def test_handle_drop_valid_image(app):
    """Test handle_drop with a valid image path."""
    event = MagicMock()
    event.data = "{C:/path with space/test.png}"
    
    app.load_image = MagicMock()
    app.handle_drop(event)
    
    app.load_image.assert_called_with("C:/path with space/test.png")

def test_handle_drop_invalid_file(app):
    """Test handle_drop with an invalid file type."""
    event = MagicMock()
    event.data = "test.txt"
    
    # Importing here to ensure we patch the right reference
    import Image_upscaler
    with patch('Image_upscaler.messagebox.showwarning') as mock_warning:
        app.load_image = MagicMock()
        app.handle_drop(event)
        
        app.load_image.assert_not_called()
        mock_warning.assert_called_once()
