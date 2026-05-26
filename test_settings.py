import pytest
from unittest.mock import MagicMock, patch
import sys
import os
import json

# Define dummy base classes to avoid metaclass conflict during testing
class DummyCTk:
    def __init__(self, *args, **kwargs): pass
    def grid_columnconfigure(self, *args, **kwargs): pass
    def grid_rowconfigure(self, *args, **kwargs): pass
    def title(self, *args, **kwargs): pass
    def geometry(self, *args, **kwargs): pass
    def resizable(self, *args, **kwargs): pass
    def attributes(self, *args, **kwargs): pass
    def after(self, *args, **kwargs): pass
    def mainloop(self, *args, **kwargs): pass
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

# Setup global persistent mocks
mock_ctk = MagicMock()
mock_ctk.CTk = DummyCTk
mock_ctk.CTkFont = MagicMock()
mock_ctk.ThemeManager.theme = {"CTkLabel": {"text_color": "white"}}

# Setup individual structural components to satisfy pack and widget assignment chains
mock_element = MagicMock()
mock_ctk.CTkFrame.return_value = mock_element
mock_ctk.CTkLabel.return_value = mock_element
mock_ctk.CTkCheckBox.return_value = mock_element
mock_ctk.CTkEntry.return_value = mock_element
mock_ctk.CTkButton.return_value = mock_element

mock_dnd = MagicMock()
mock_dnd.TkinterDnD.DnDWrapper = DummyDnDWrapper
mock_dnd.DND_ALL = "all"

sys.modules['customtkinter'] = mock_ctk
sys.modules['ctk'] = mock_ctk
sys.modules['tkinterdnd2'] = mock_dnd
sys.modules['torch'] = MagicMock()
sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.transforms'] = MagicMock()
sys.modules['spandrel'] = MagicMock()

from Image_upscaler import Image_upscaler

@pytest.fixture
def app():
    with patch('Image_upscaler.UpscalerEngine'), \
         patch('Image_upscaler.TkinterDnD._require'), \
         patch('Image_upscaler.Image_upscaler._load_settings'):
        with patch('customtkinter.CTkFrame'), \
             patch('customtkinter.CTkLabel'), \
             patch('customtkinter.CTkButton'), \
             patch('customtkinter.CTkOptionMenu'), \
             patch('customtkinter.CTkProgressBar'), \
             patch('psutil.Process'):
            
            app_instance = Image_upscaler()
            
            # Reset default values to ensure clean test state
            app_instance.autosave_enabled = False
            app_instance.autosave_dir = ""
            
            return app_instance

def test_settings_initial_state(app):
    """Verify that settings are initialized to default values."""
    assert hasattr(app, 'autosave_enabled')
    assert app.autosave_enabled is False
    assert hasattr(app, 'autosave_dir')
    assert app.autosave_dir == ""

def test_open_settings(app):
    """Test that open_settings builds the Toplevel frame and invokes UI state updates."""
    app.settings_window = None
    spy_toplevel = MagicMock()
    
    with patch('Image_upscaler.ctk.CTkToplevel', return_value=spy_toplevel) as mock_create:
        app.open_settings()
        mock_create.assert_called_once_with(app)

def test_autosave_logic_in_finish(app):
    """Test that _finish attempts to autosave if enabled."""
    app.autosave_enabled = True
    app.autosave_dir = "/fake/dir"
    app.output_image = MagicMock()
    app.input_path = "test.png"
    
    with patch('os.path.exists', return_value=True), \
         patch('Image_upscaler.messagebox.showinfo'):
        app._finish(MagicMock(), "test.png")
        assert app.output_image.save.called

def test_load_settings_corrupted(app, tmp_path):
    """Test that _load_settings handles corrupted JSON files gracefully."""
    settings_file = tmp_path / "settings.json"
    settings_file.write_text("{corrupted: json")
    
    with patch('Image_upscaler.os.path.join', return_value=str(settings_file)):
        # Should not raise exception
        app._load_settings()
        assert app.autosave_enabled is False

def test_save_settings_validation_invalid_dir(app):
    """Test that _save_settings validates directory presence when enabled."""
    app.check_autosave = MagicMock()
    app.check_autosave.get.return_value = True
    app.entry_dir = MagicMock()
    app.entry_dir.get.return_value = "" # Empty dir
    
    with patch('Image_upscaler.messagebox.showerror') as mock_error:
        app._save_settings()
        mock_error.assert_called_with("Settings Error", "Please select a directory for autosave.")
        assert app.autosave_enabled is False

def test_save_settings_validation_unwritable_dir(app):
    """Test that _save_settings validates directory writability."""
    app.check_autosave = MagicMock()
    app.check_autosave.get.return_value = True
    app.entry_dir = MagicMock()
    app.entry_dir.get.return_value = "/read-only-dir"
    
    app.settings_window = MagicMock()
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', side_effect=PermissionError("Permission denied")), \
         patch('Image_upscaler.messagebox.showerror') as mock_error:
        app._save_settings()
        mock_error.assert_called()
        assert "Cannot use this directory" in mock_error.call_args[0][1]
        assert app.autosave_enabled is False

def test_autosave_counter_logic(app, tmp_path):
    """Test that autosave handles filename collisions using a counter."""
    app.autosave_enabled = True
    app.autosave_dir = str(tmp_path)
    app.output_image = MagicMock()
    app.input_path = "test.png"
    
    # Create existing files to trigger counter
    (tmp_path / "test_upscaled.png").touch()
    (tmp_path / "test_upscaled_1.png").touch()
    
    with patch('Image_upscaler.messagebox.showinfo'):
        app._finish(MagicMock(), "test.png")
        
        expected_path = os.path.join(str(tmp_path), "test_upscaled_2.png")
        app.output_image.save.assert_called_with(expected_path)
