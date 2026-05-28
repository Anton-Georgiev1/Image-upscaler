import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Define dummy base classes to avoid metaclass conflict during testing
class DummyCTk:
    def __init__(self, *args, **kwargs): pass
    def grid_columnconfigure(self, *args, **kwargs): pass
    def grid_rowconfigure(self, *args, **kwargs): pass
    def title(self, *args, **kwargs): pass
    def geometry(self, *args, **kwargs): pass
    def after(self, *args, **kwargs): pass
    def protocol(self, *args, **kwargs): pass
    def destroy(self, *args, **kwargs): pass
    def winfo_exists(self, *args, **kwargs): return True
    def splitlist(self, data):
        if not data: return []
        if data.startswith('{') and data.endswith('}'):
            return [data[1:-1]]
        return data.split()

class DummyDnDWrapper:
    def __init__(self, *args, **kwargs): pass
    def drop_target_register(self, *args, **kwargs): pass
    def dnd_bind(self, *args, **kwargs): pass

# Mock the modules before importing Image_upscaler
mock_ctk = MagicMock()
mock_ctk.CTk = DummyCTk
mock_ctk.CTkFrame = MagicMock()
mock_ctk.CTkLabel = MagicMock()
mock_ctk.CTkEntry = MagicMock(side_effect=lambda *args, **kwargs: MagicMock())
mock_ctk.CTkButton = MagicMock()
mock_ctk.CTkOptionMenu = MagicMock()
mock_ctk.CTkProgressBar = MagicMock()
mock_ctk.CTkFont = MagicMock()
mock_ctk.ThemeManager.theme = {"CTkLabel": {"text_color": "white"}}

mock_dnd = MagicMock()
mock_dnd.TkinterDnD.DnDWrapper = DummyDnDWrapper
mock_dnd.DND_ALL = "all"

sys.modules['customtkinter'] = mock_ctk
sys.modules['tkinterdnd2'] = mock_dnd
sys.modules['torch'] = MagicMock()
sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.transforms'] = MagicMock()
sys.modules['spandrel'] = MagicMock()

from Image_upscaler import Image_upscaler

@pytest.fixture
def app():
    with patch('Image_upscaler.UpscalerEngine'), \
         patch('Image_upscaler.TkinterDnD._require'):
        with patch('customtkinter.CTkFrame'), \
             patch('customtkinter.CTkLabel'), \
             patch('customtkinter.CTkButton'), \
             patch('customtkinter.CTkOptionMenu'), \
             patch('customtkinter.CTkProgressBar'):
            return Image_upscaler()

def test_load_image(app):
    app.btn_run = MagicMock()
    app.btn_save = MagicMock()
    app.show_preview = MagicMock()
    
    test_path = "test.png"
    app.load_image(test_path)

    assert app.input_path == os.path.normpath(test_path)
    app.btn_run.configure.assert_called_with(state="normal")
    app.btn_save.configure.assert_called_with(state="disabled")
    app.show_preview.assert_called_with(os.path.normpath(test_path))

def test_handle_drop_valid_image(app):
    event = MagicMock()
    event.data = "{C:/path with space/test.png}"

    with patch('os.path.isfile', return_value=True):
        app.load_image = MagicMock()
        app.handle_drop(event)
        
        app.load_image.assert_called_with(os.path.normpath("C:/path with space/test.png"))

def test_handle_drop_during_processing(app):
    """Test that handle_drop ignores drops while processing."""
    app.is_processing = True
    event = MagicMock()
    event.data = "test.png"
    
    app.load_image = MagicMock()
    app.handle_drop(event)
    
    app.load_image.assert_not_called()

def test_run_upscale_sets_processing(app):
    """Test that run_upscale sets is_processing to True."""
    app.input_path = "test.png"
    app.opt_model.get.return_value = "Lanczos (Fast CPU)"
    app.opt_scale.get.return_value = "2x"
    app.opt_perf.get.return_value = "Ultra (Max Speed)"
    
    with patch('os.path.exists', return_value=True), \
         patch('threading.Thread'):
        app.run_upscale()
        assert app.is_processing is True

def test_finish_resets_processing(app):
    """Test that _finish sets is_processing to False."""
    app.is_processing = True
    app.output_image = MagicMock()
    
    with patch('Image_upscaler.messagebox.showinfo'), \
         patch('Image_upscaler.os.path.exists', return_value=True):
        app._finish(MagicMock(), "test.png")
        assert app.is_processing is False

def test_handle_drop_folder(app):
    event = MagicMock()
    event.data = "C:/some_folder"

    with patch('os.path.isfile', return_value=False), \
         patch('Image_upscaler.messagebox.showwarning') as mock_warning:
        app.load_image = MagicMock()
        app.handle_drop(event)
        
        app.load_image.assert_not_called()
        mock_warning.assert_called_once_with("Invalid Drop", "Please drop a file, not a folder.")

def test_swap_custom_size(app):
    """Test that swap_custom_size exchanges width and height values."""
    app.entry_width.get.return_value = "100"
    app.entry_height.get.return_value = "200"
    
    app.swap_custom_size()
    
    app.entry_width.delete.assert_called_with(0, "end")
    app.entry_width.insert.assert_called_with(0, "200")
    app.entry_height.delete.assert_called_with(0, "end")
    app.entry_height.insert.assert_called_with(0, "100")
