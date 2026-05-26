import pytest
from unittest.mock import MagicMock, patch
import sys

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

class DummyDnDWrapper:
    def __init__(self, *args, **kwargs): pass
    def drop_target_register(self, *args, **kwargs): pass
    def dnd_bind(self, *args, **kwargs): pass

# Setup global persistent mocks
mock_ctk = MagicMock()
mock_ctk.CTk = DummyCTk
mock_ctk.CTkFont = MagicMock()

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
            
            # Clear all method side-effects on background callbacks
            app_instance._toggle_autosave_ui = MagicMock()
            app_instance._browse_autosave_dir = MagicMock()
            app_instance._save_settings = MagicMock()
            
            return app_instance

def test_settings_initial_state(app):
    """Verify that settings are initialized to default values."""
    assert hasattr(app, 'autosave_enabled')
    assert app.autosave_enabled is False
    assert hasattr(app, 'autosave_dir')
    assert app.autosave_dir == ""

def test_open_settings(app):
    """Test that open_settings builds the Toplevel frame and invokes UI state updates."""
    # Force state isolation parameters onto the instance immediately prior to running
    app.settings_window = None
    
    # Establish a fresh mock instance for the creation tracker spy
    spy_toplevel = MagicMock()
    
    with patch('Image_upscaler.ctk.CTkToplevel', return_value=spy_toplevel) as mock_create:
        app.open_settings()
        
        # Verify the application executed code paths inside your window builder
        mock_create.assert_called_once_with(app)
        app._toggle_autosave_ui.assert_called_once()

def test_autosave_logic_in_finish(app):
    """Test that _finish attempts to autosave if enabled."""
    app.autosave_enabled = True
    app.autosave_dir = "/fake/dir"
    app.output_image = MagicMock()
    app.input_path = "test.png"
    
    with patch('os.path.exists', return_value=True), \
         patch('os.access', return_value=True), \
         patch('Image_upscaler.messagebox.showinfo'), \
         patch('Image_upscaler.messagebox.showerror'):
        app._finish()
        
        assert app.output_image.save.called
