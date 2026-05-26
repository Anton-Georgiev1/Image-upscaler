import customtkinter as ctk
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_ALL
from PIL import Image
import os
import json
import threading
import math
import psutil
from pathlib import Path

# --- Modern AI Engine Integration (Spandrel) ---
# Removed try/except so the app explicitly requires AI packages to run
import torch
from torchvision.transforms import ToTensor, ToPILImage
from spandrel import ModelLoader

HAS_AI = True

class UpscalerEngine:
    """Core logic for Real-ESRGAN using modern Spandrel."""
    
    def __init__(self):
        self.has_ai = HAS_AI
        self.ai_model = None
        # Safely assign GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_ai_model(self):
        """Loads and caches the RealESRGAN model directly."""
        if self.ai_model is not None:
            return self.ai_model
            
        weight_path = "models/RealESRGAN_x4plus.pth"
        
        # Since the auto-download is removed, we explicitly check for the file
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"Model weights not found!\n\n"
                f"Please place 'RealESRGAN_x4plus.pth' inside the '{os.path.abspath('models')}' folder."
            )
            
        # Spandrel natively reads the architecture from the .pth file
        model = ModelLoader().load_from_file(weight_path)
        self.ai_model = model.to(self.device).eval()
        return self.ai_model

    def _process_with_tiling(self, img_tensor, model, scale=4, tile_size=256, tile_pad=16, progress_callback=None, perf_mode="Responsive"):
        """Processes the image in small chunks to prevent VRAM crashes on 4K bounds."""
        b, c, h, w = img_tensor.shape
        out_h, out_w = h * scale, w * scale
        
        # Optimization: Limit threads in Responsive mode to keep system smooth
        if self.device.type == "cpu":
            if "Responsive" in perf_mode:
                torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
            else:
                torch.set_num_threads(os.cpu_count() or 1)

        # Create out_tensor on CPU to save VRAM for the AI model processing
        out_tensor = torch.zeros((b, c, out_h, out_w), device="cpu")
        
        num_tiles_y = math.ceil(h / tile_size)
        num_tiles_x = math.ceil(w / tile_size)
        total_tiles = max(1, num_tiles_y * num_tiles_x)
        processed_tiles = 0

        # Use inference_mode for better performance and less memory usage
        with torch.inference_mode():
            for y in range(0, h, tile_size):
                for x in range(0, w, tile_size):
                    y_start = max(0, y - tile_pad)
                    x_start = max(0, x - tile_pad)
                    y_end = min(h, y + tile_size + tile_pad)
                    x_end = min(w, x + tile_size + tile_pad)
                    
                    in_tile = img_tensor[:, :, y_start:y_end, x_start:x_end]
                    
                    if self.device.type == "cuda":
                        with torch.autocast(device_type="cuda"):
                            out_tile = model(in_tile)
                    else:
                        out_tile = model(in_tile)
                    
                    tile_y_offset = (y - y_start) * scale
                    tile_x_offset = (x - x_start) * scale
                    
                    valid_h = min(tile_size, h - y) * scale
                    valid_w = min(tile_size, w - x) * scale
                    
                    valid_out_tile = out_tile[:, :, 
                                            tile_y_offset : tile_y_offset + valid_h, 
                                            tile_x_offset : tile_x_offset + valid_w]
                    
                    out_y, out_x = y * scale, x * scale
                    out_tensor[:, :, out_y : out_y + valid_h, out_x : out_x + valid_w] = valid_out_tile.cpu()
                    
                    processed_tiles += 1
                    if progress_callback:
                        progress_callback(processed_tiles / total_tiles)
                
        return out_tensor

    def _upscale_ai(self, path, progress_callback=None, perf_mode="Responsive"):
        """Converts to tensor, runs PyTorch inference, converts back to Image. Preserves Alpha."""
        img = Image.open(path)
        has_alpha = img.mode == "RGBA"
        
        if has_alpha:
            alpha = img.getchannel("A")
            img = img.convert("RGB")
        else:
            img = img.convert("RGB")

        img_tensor = ToTensor()(img).unsqueeze(0).to(self.device)
        
        model = self._init_ai_model()
        scale = model.scale
        
        out_tensor = self._process_with_tiling(img_tensor, model, scale=scale, progress_callback=progress_callback, perf_mode=perf_mode)
        out_img = ToPILImage()(out_tensor.squeeze(0).clamp(0, 1))
        
        if has_alpha:
            # Upscale alpha channel using Lanczos for consistency
            new_size = out_img.size
            alpha_upscaled = alpha.resize(new_size, Image.Resampling.LANCZOS)
            out_img.putalpha(alpha_upscaled)

        # Free up memory safely
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            
        return out_img

    def upscale(self, input_path: str, model_type: str, scale_val: str, progress_callback=None, perf_mode="Responsive", custom_size=None) -> Image.Image:
        if not Path(input_path).exists():
            raise ValueError("Input file not found.")

        # 1. Parse User Intent
        is_4k_mode = (scale_val == "4K (Ultra HD)")
        is_custom_mode = (scale_val == "Custom Size")
        
        img = Image.open(input_path)
        w, h = img.size
        
        # 2. Determine Scale Factors and Target Dimensions
        target_w, target_h = None, None

        if is_4k_mode:
            # 4K logic: Fit within 3840x2160 box
            target_w_box, target_h_box = 3840, 2160
            fit_ratio = min(target_w_box / w, target_h_box / h)
            target_scale = 1.0 if fit_ratio <= 1.0 else fit_ratio
        elif is_custom_mode and custom_size:
            target_w, target_h = custom_size
            # Calculate required scale to reach the larger dimension
            target_scale = max(target_w / w, target_h / h)
        else:
            # Multiplier logic: 2x, 3x, 4x
            target_scale = float(scale_val.replace("x", ""))

        # 3. Execution Phase
        if target_scale <= 1.0 and not is_custom_mode:
            if progress_callback: progress_callback(1.0)
            result = img
        elif self.has_ai and "RealESRGAN" in model_type:
            # AI always outputs 4x (native model scale)
            img.close()
            result = self._upscale_ai(input_path, progress_callback=progress_callback, perf_mode=perf_mode)
            
            # Post-processing: Resize AI 4x result to match User Intent
            if is_4k_mode:
                result = self._resize_to_target(result, 3840, 2160)
            elif is_custom_mode:
                # User specified exact pixels
                result = result.resize((target_w, target_h), Image.Resampling.LANCZOS)
            elif target_scale != 4.0:
                # Downsample 4x AI result to 2x or 3x for high quality (Supersampling)
                # or upsample if user somehow picked > 4x
                result = result.resize((int(w * target_scale), int(h * target_scale)), Image.Resampling.LANCZOS)
        else:
            # CPU/PIL Path
            img.close()
            if progress_callback: progress_callback(0.5)
            # We scale to the required multiplier using high-quality Lanczos
            result = self._upscale_pil(input_path, target_scale)
            
            if is_4k_mode:
                result = self._resize_to_target(result, 3840, 2160)
            elif is_custom_mode:
                result = result.resize((target_w, target_h), Image.Resampling.LANCZOS)
            
            if progress_callback: progress_callback(1.0)

        return result

    def _resize_to_target(self, img: Image.Image, target_w: int, target_h: int) -> Image.Image:
        """Lock into exact target dimensions without losing aspect ratio."""
        w, h = img.size
        ratio = min(target_w / w, target_h / h)
        new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
        return img.resize(new_size, Image.Resampling.LANCZOS)

    def _upscale_pil(self, path, scale):
        img = Image.open(path)
        new_size = (int(img.width * scale), int(img.height * scale))
        return img.resize(new_size, Image.Resampling.LANCZOS)


# --- GUI Application ---
class Image_upscaler(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        super().__init__()
        
        # Initialize TkinterDnD hooks
        self.TkdndVersion = TkinterDnD._require(self)

        # Renamed application title
        self.title("Image_upscaler")
        self.geometry("1100x750")
        
        self.engine = UpscalerEngine()
        self.input_path = None
        self.output_image = None
        
        # Application state
        self.is_processing = False
        
        # Settings state
        self.autosave_enabled = False
        self.autosave_dir = ""
        self.settings_window = None
        self._load_settings()

        # --- Layout ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        # Renamed sidebar title
        self.lbl_title = ctk.CTkLabel(self.sidebar, text="Image_upscaler", font=ctk.CTkFont(size=22, weight="bold"))
        self.lbl_title.pack(pady=(30, 20))

        self.btn_select = ctk.CTkButton(self.sidebar, text="Open Image", command=self.select_image, height=40)
        self.btn_select.pack(padx=20, pady=10, fill="x")

        self.lbl_model = ctk.CTkLabel(self.sidebar, text="Upscale Model:", anchor="w")
        self.lbl_model.pack(padx=20, pady=(20, 0), fill="x")
        
        model_options = ["RealESRGAN (AI HD)", "Lanczos (Fast CPU)"]
            
        self.opt_model = ctk.CTkOptionMenu(self.sidebar, values=model_options)
        self.opt_model.pack(padx=20, pady=10, fill="x")

        self.lbl_scale = ctk.CTkLabel(self.sidebar, text="Scale Factor:", anchor="w")
        self.lbl_scale.pack(padx=20, pady=(10, 0), fill="x")
        
        self.opt_scale = ctk.CTkOptionMenu(self.sidebar, values=["2x", "3x", "4x", "4K (Ultra HD)", "Custom Size"], command=self._on_scale_change)
        self.opt_scale.pack(padx=20, pady=10, fill="x")

        # Custom Size Inputs (hidden initially)
        self.frame_custom = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        
        self.entry_width = ctk.CTkEntry(self.frame_custom, placeholder_text="Width", width=100)
        self.entry_width.pack(side="left", padx=(20, 5), pady=5)
        
        self.lbl_x = ctk.CTkLabel(self.frame_custom, text="x")
        self.lbl_x.pack(side="left", pady=5)
        
        self.entry_height = ctk.CTkEntry(self.frame_custom, placeholder_text="Height", width=100)
        self.entry_height.pack(side="left", padx=(5, 20), pady=5)

        # Ensure frame_custom is in the right packing order even when hidden
        self.frame_custom.pack(padx=20, pady=5, fill="x")
        self.frame_custom.pack_forget()

        self.lbl_perf = ctk.CTkLabel(self.sidebar, text="Processing Mode:", anchor="w")
        self.lbl_perf.pack(padx=20, pady=(10, 0), fill="x")
        
        self.opt_perf = ctk.CTkOptionMenu(self.sidebar, values=["Responsive (Smooth PC)", "Ultra (Max Speed)"])
        self.opt_perf.pack(padx=20, pady=10, fill="x")

        self.btn_run = ctk.CTkButton(self.sidebar, text="Start Upscale", command=self.run_upscale, 
                                     fg_color="#1f6aa5", hover_color="#144870", height=45, state="disabled")
        self.btn_run.pack(padx=20, pady=(40, 10), fill="x")

        self.btn_save = ctk.CTkButton(self.sidebar, text="Save Result", command=self.save_image, state="disabled", height=40)
        self.btn_save.pack(padx=20, pady=10, fill="x")

        self.btn_settings = ctk.CTkButton(self.sidebar, text="Settings", command=self.open_settings, 
                                          fg_color="transparent", border_width=2, height=40)
        self.btn_settings.pack(padx=20, pady=10, fill="x")

        # Theme toggle
        self.lbl_theme = ctk.CTkLabel(self.sidebar, text="Theme:", anchor="w")
        self.lbl_theme.pack(padx=20, pady=(20, 0), fill="x", side="bottom")
        self.opt_theme = ctk.CTkOptionMenu(self.sidebar, values=["Dark", "Light"], command=lambda m: ctk.set_appearance_mode(m))
        self.opt_theme.pack(padx=20, pady=(5, 30), fill="x", side="bottom")

        # System Monitor
        self.process = psutil.Process(os.getpid())
        self.lbl_cpu = ctk.CTkLabel(self.sidebar, text="System Load: 0%", anchor="w", font=ctk.CTkFont(size=12))
        self.lbl_cpu.pack(padx=20, pady=(10, 0), fill="x", side="bottom")
        
        device_name = "GPU (CUDA)" if self.engine.device.type == "cuda" else "CPU (Standard)"
        self.lbl_device = ctk.CTkLabel(self.sidebar, text=f"Active Device: {device_name}", anchor="w", font=ctk.CTkFont(size=11), text_color="gray")
        self.lbl_device.pack(padx=20, pady=(0, 30), fill="x", side="bottom")
        
        self._update_cpu_usage()

        # Main View
        self.main_view = ctk.CTkFrame(self, corner_radius=15)
        self.main_view.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_view.grid_rowconfigure(0, weight=1)
        self.main_view.grid_columnconfigure(0, weight=1)

        self.lbl_preview = ctk.CTkLabel(self.main_view, text="Select or Drag & Drop an image to preview", text_color="gray")
        self.lbl_preview.grid(row=0, column=0, sticky="nsew")
        
        # Enable Drag & Drop on the preview label
        self.lbl_preview.drop_target_register(DND_ALL)
        self.lbl_preview.dnd_bind('<<Drop>>', self.handle_drop)

        self.progress = ctk.CTkProgressBar(self.main_view)
        self.progress.grid(row=1, column=0, padx=50, pady=(0, 5), sticky="ew")
        self.progress.set(0)

        self.lbl_status = ctk.CTkLabel(self.main_view, text="Ready", font=ctk.CTkFont(size=12))
        self.lbl_status.grid(row=2, column=0, pady=(0, 20))

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")])
        if path:
            self.load_image(path)

    def handle_drop(self, event):
        if self.is_processing:
            return

        # Robust path parsing for TkinterDnD (handles spaces and braces)
        try:
            paths = self.splitlist(event.data)
        except Exception:
            # Fallback for unexpected data formats
            import re
            data = event.data
            paths = re.findall(r'\{(.*?)\}|(\S+)', data)
            paths = [p[0] if p[0] else p[1] for p in paths]
        
        if not paths: return
        
        path = paths[0]
        # Normalize and check if it's a file
        path = os.path.normpath(path)
        if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
            self.load_image(path)
        elif not os.path.isfile(path):
             messagebox.showwarning("Invalid Drop", "Please drop a file, not a folder.")
        else:
            messagebox.showwarning("Invalid File", "Please drop a valid image file (PNG, JPG, WEBP, BMP).")

    def load_image(self, path):
        # Normalize path
        path = os.path.normpath(path)
        self.input_path = path
        self.btn_run.configure(state="normal")
        self.btn_save.configure(state="disabled")
        self.show_preview(path)

    def show_preview(self, path_or_img):
        try:
            if isinstance(path_or_img, str):
                img = Image.open(path_or_img).convert("RGB")
            else:
                img = path_or_img.copy()
                
            target_w, target_h = 750, 550
            img.thumbnail((target_w, target_h))
            
            self.preview_tk = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            self.lbl_preview.configure(image=self.preview_tk, text="")
        except Exception as e:
            messagebox.showerror("Preview Error", f"Failed to load preview: {e}")

    def _on_scale_change(self, choice):
        if choice == "Custom Size":
            self.frame_custom.pack(pady=5, fill="x", after=self.opt_scale)
        else:
            self.frame_custom.pack_forget()

    def run_upscale(self):
        if not self.input_path: return
        
        # Check if input file still exists
        if not os.path.exists(self.input_path):
            messagebox.showerror("File Error", "The input image file no longer exists.")
            return

        model = self.opt_model.get()
        scale_val = self.opt_scale.get() 
        perf_mode = self.opt_perf.get()
        
        custom_size = None
        if scale_val == "Custom Size":
            try:
                cw = int(self.entry_width.get())
                ch = int(self.entry_height.get())
                if cw <= 0 or ch <= 0: raise ValueError()
                custom_size = (cw, ch)
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid positive numbers for Width and Height.")
                return

        # Checking for the manual weights file if AI is selected
        if "RealESRGAN" in model and not os.path.exists("models/RealESRGAN_x4plus.pth"):
            messagebox.showerror(
                "Missing Model File", 
                "The Real-ESRGAN weights were not found.\n\nPlease place 'RealESRGAN_x4plus.pth' in the 'models' folder next to the script."
            )
            return

        self.btn_run.configure(state="disabled")
        self.btn_select.configure(state="disabled")
        self.btn_save.configure(state="disabled")
        self.is_processing = True
        
        self.progress.set(0)
        self.lbl_status.configure(text="Initializing AI Engine...")
        
        # Optimization: Set process priority based on mode
        try:
            if "Responsive" in perf_mode:
                self.process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                self.process.nice(psutil.NORMAL_PRIORITY_CLASS)
        except:
            pass

        def progress_callback(pct):
            # Use self.after to update UI from the background thread safely
            self.after(0, lambda: self._update_progress_ui(pct))

        # Capture input_path locally to avoid race conditions if self.input_path changes
        current_input_path = self.input_path
        threading.Thread(target=self._process, args=(current_input_path, model, scale_val, progress_callback, perf_mode, custom_size), daemon=True).start()

    def _update_progress_ui(self, pct):
        self.progress.set(pct)
        self.lbl_status.configure(text=f"Processing: {int(pct * 100)}%")

    def _process(self, input_path, model, scale_val, progress_callback, perf_mode, custom_size):
        try:
            result = self.engine.upscale(input_path, model, scale_val, 
                                         progress_callback=progress_callback, 
                                         perf_mode=perf_mode, 
                                         custom_size=custom_size)
            self.after(0, lambda: self._finish(result, input_path))
        except Exception as e:
            # Enhanced error reporting
            err_msg = str(e)
            if "out of memory" in err_msg.lower():
                err_msg = "GPU Out of Memory! Try a smaller scale or check if other apps are using VRAM."
            
            self.after(0, lambda: messagebox.showerror("Upscale Error", f"Failed: {err_msg}"))
            self.after(0, self._reset_ui)

    def _finish(self, result, input_path):
        self.output_image = result
        self.is_processing = False
        
        # Restore normal priority
        try:
            self.process.nice(psutil.NORMAL_PRIORITY_CLASS)
        except:
            pass
            
        self.progress.set(1)
        self.lbl_status.configure(text="Complete! (100%)")
        self.show_preview(self.output_image)
        self.btn_run.configure(state="normal")
        self.btn_select.configure(state="normal")
        self.btn_save.configure(state="normal")

        autosave_info = ""
        if self.autosave_enabled and self.autosave_dir:
            try:
                if not os.path.exists(self.autosave_dir):
                    os.makedirs(self.autosave_dir, exist_ok=True)
                
                input_name = Path(input_path).stem
                output_name = f"{input_name}_upscaled.png"
                save_path = os.path.join(self.autosave_dir, output_name)
                
                counter = 1
                while os.path.exists(save_path) and counter < 1000:
                    save_path = os.path.join(self.autosave_dir, f"{input_name}_upscaled_{counter}.png")
                    counter += 1
                
                self.output_image.save(save_path)
                autosave_info = f"\n\nAutosaved to: {save_path}"
            except Exception as e:
                messagebox.showerror("Autosave Error", f"Failed to autosave image: {e}")

        messagebox.showinfo("Success", f"Upscaling complete!{autosave_info}")

    def _reset_ui(self):
        self.is_processing = False
        # Restore normal priority
        try:
            self.process.nice(psutil.NORMAL_PRIORITY_CLASS)
        except:
            pass
            
        self.progress.set(0)
        self.lbl_status.configure(text="Ready")
        self.btn_run.configure(state="normal")
        self.btn_select.configure(state="normal")

    def save_image(self):
        if not self.output_image: return
        path = filedialog.asksaveasfilename(defaultextension=".png", 
                                            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
        if path:
            try:
                self.output_image.save(path)
                messagebox.showinfo("Saved", f"Image saved successfully to:\n{path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save image: {e}")

    def _load_settings(self):
        """Loads settings from a JSON file."""
        settings_path = os.path.join(os.path.dirname(__file__), "settings.json")
        if os.path.exists(settings_path):
            try:
                with open(settings_path, "r") as f:
                    settings = json.load(f)
                    if isinstance(settings, dict):
                        self.autosave_enabled = settings.get("autosave_enabled", False)
                        self.autosave_dir = settings.get("autosave_dir", "")
            except Exception as e:
                print(f"Error loading settings: {e}")

    def _save_settings_to_disk(self):
        """Saves current settings to a JSON file."""
        settings_path = os.path.join(os.path.dirname(__file__), "settings.json")
        settings = {
            "autosave_enabled": self.autosave_enabled,
            "autosave_dir": self.autosave_dir
        }
        try:
            with open(settings_path, "w") as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            messagebox.showerror("Settings Error", f"Failed to save settings to disk: {e}")

    def open_settings(self):
        """Opens a window for application settings."""
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.focus()
            return

        self.settings_window = ctk.CTkToplevel(self)
        self.settings_window.title("Settings")
        self.settings_window.geometry("500x400")
        self.settings_window.resizable(False, False)
        
        # Ensure it stays on top and grabs focus
        self.settings_window.attributes("-topmost", True)
        self.settings_window.after(100, self.settings_window.focus)
        
        main_frame = ctk.CTkFrame(self.settings_window, corner_radius=10)
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        lbl_title = ctk.CTkLabel(main_frame, text="Application Settings", font=ctk.CTkFont(size=18, weight="bold"))
        lbl_title.pack(pady=(15, 20))

        # Autosave Toggle
        self.check_autosave = ctk.CTkCheckBox(main_frame, text="Enable Autosave after upscaling", 
                                              command=self._toggle_autosave_ui)
        if self.autosave_enabled:
            self.check_autosave.select()
        self.check_autosave.pack(padx=20, pady=10, anchor="w")

        # Save Directory Selection
        self.frame_dir = ctk.CTkFrame(main_frame, fg_color="transparent")
        self.frame_dir.pack(padx=20, pady=5, fill="x")

        self.lbl_dir_title = ctk.CTkLabel(self.frame_dir, text="Autosave Directory:", font=ctk.CTkFont(size=12))
        self.lbl_dir_title.pack(anchor="w")

        self.entry_dir = ctk.CTkEntry(self.frame_dir, placeholder_text="Select directory...", width=300)
        self.entry_dir.insert(0, self.autosave_dir)
        self.entry_dir.pack(side="left", pady=5, fill="x", expand=True)
        
        self.btn_browse = ctk.CTkButton(self.frame_dir, text="Browse", width=80, command=self._browse_autosave_dir)
        self.btn_browse.pack(side="left", padx=(10, 0), pady=5)

        # Buttons Frame
        btn_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        btn_frame.pack(pady=(30, 10), fill="x")

        self.btn_save_settings = ctk.CTkButton(btn_frame, text="Save Settings", command=self._save_settings,
                                               fg_color="#1f6aa5", hover_color="#144870")
        self.btn_save_settings.pack(side="right", padx=10)

        btn_cancel = ctk.CTkButton(btn_frame, text="Cancel", command=self.settings_window.destroy,
                                   fg_color="transparent", border_width=1)
        btn_cancel.pack(side="right", padx=10)

        # Initialize UI state
        self._toggle_autosave_ui()

    def _save_settings(self):
        """Validates and saves settings."""
        is_enabled = bool(self.check_autosave.get())
        new_dir = self.entry_dir.get().strip()
        
        if is_enabled:
            if not new_dir:
                messagebox.showerror("Settings Error", "Please select a directory for autosave.")
                return
            
            # Check if directory is valid/writable
            try:
                if not os.path.exists(new_dir):
                    # Try creating it to verify validity
                    os.makedirs(new_dir, exist_ok=True)
                
                # Verify writability
                test_file = os.path.join(new_dir, ".write_test")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                messagebox.showerror("Settings Error", f"Cannot use this directory:\n{e}")
                return

        self.autosave_enabled = is_enabled
        self.autosave_dir = new_dir
        self._save_settings_to_disk()
        self.settings_window.destroy()
        messagebox.showinfo("Settings", "Settings saved successfully.")

    def _toggle_autosave_ui(self):
        is_checked = self.check_autosave.get()
        state = "normal" if is_checked else "disabled"
        self.entry_dir.configure(state=state)
        self.btn_browse.configure(state=state)
        
        # Adjust label color based on state for better visibility
        if is_checked:
            self.lbl_dir_title.configure(text_color=ctk.ThemeManager.theme["CTkLabel"]["text_color"]) # Reset to theme default
        else:
            self.lbl_dir_title.configure(text_color="gray")

    def _browse_autosave_dir(self):
        current_dir = self.entry_dir.get().strip() or os.getcwd()
        directory = filedialog.askdirectory(initialdir=current_dir)
        if directory:
            self.entry_dir.delete(0, "end")
            self.entry_dir.insert(0, directory)

    def _update_cpu_usage(self):
        """Periodically updates the CPU usage label."""
        try:
            # interval=None is non-blocking and returns usage since last call
            cpu_pct = psutil.cpu_percent(interval=None)
            self.lbl_cpu.configure(text=f"System Load: {int(cpu_pct)}%")
        except:
            pass
        # Refresh every 1000ms
        self.after(1000, self._update_cpu_usage)

if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    app = Image_upscaler()
    app.mainloop()