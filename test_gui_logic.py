import pytest
from unittest.mock import MagicMock, patch
import sys

import os

class DummyCTk:
    def __init__(self, *args, **kwargs): pass
    def grid_columnconfigure(self, *args, **kwargs): pass
    def grid_rowconfigure(self, *args, **kwargs): pass
    def title(self, *args, **kwargs): pass
    def geometry(self, *args, **kwargs): pass
    def after(self, *args, **kwargs): pass
    def splitlist(self, data):
        if not data: return []
        if data.startswith('{') and data.endswith('}'):
            return [data[1:-1]]
        return data.split()

class DummyDnDWrapper:
    def __init__(self, *args, **kwargs): pass
    def drop_target_register(self, *args, **kwargs): pass
    def dnd_bind(self, *args, **kwargs): pass
    def splitlist(self, data):
        return DummyCTk.splitlist(self, data)

mock_ctk = MagicMock()
mock_ctk.CTk = DummyCTk
mock_ctk.CTkFrame = MagicMock()
mock_ctk.CTkLabel = MagicMock()
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

def test_handle_drop_nested_braces(app):
    event = MagicMock()
    # Mocking splitlist behavior for nested braces which standard splitlist handles
    event.data = "{C:/path {with} braces/test.png}"
    
    with patch.object(app, 'splitlist', return_value=["C:/path {with} braces/test.png"]), \
         patch('os.path.isfile', return_value=True):
        app.load_image = MagicMock()
        app.handle_drop(event)
        
        app.load_image.assert_called_with(os.path.normpath("C:/path {with} braces/test.png"))

def test_handle_drop_folder(app):
    event = MagicMock()
    event.data = "C:/some_folder"
    
    with patch('os.path.isfile', return_value=False), \
         patch('Image_upscaler.messagebox.showwarning') as mock_warning:
        app.load_image = MagicMock()
        app.handle_drop(event)
        
        app.load_image.assert_not_called()
        mock_warning.assert_called_once_with("Invalid Drop", "Please drop a file, not a folder.")

def test_handle_drop_invalid_file(app):
    event = MagicMock()
    event.data = "test.txt"
    
    with patch('os.path.isfile', return_value=True), \
         patch('Image_upscaler.messagebox.showwarning') as mock_warning:
        app.load_image = MagicMock()
        app.handle_drop(event)
        
        app.load_image.assert_not_called()
        mock_warning.assert_called_once_with("Invalid File", "Please drop a valid image file (PNG, JPG, WEBP, BMP).")
