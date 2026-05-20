import customtkinter as ctk
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_ALL
from PIL import Image
import os
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

    def _process_with_tiling(self, img_tensor, model, scale=4, tile_size=256, tile_pad=16, progress_callback=None):
        """Processes the image in small chunks to prevent VRAM crashes on 4K bounds."""
        b, c, h, w = img_tensor.shape
        out_h, out_w = h * scale, w * scale
        
        # FIX: Create out_tensor on CPU to save VRAM for the AI model processing
        # This prevents OOM when merging very large images (e.g. 16K outputs)
        out_tensor = torch.zeros((b, c, out_h, out_w), device="cpu")
        
        num_tiles_y = math.ceil(h / tile_size)
        num_tiles_x = math.ceil(w / tile_size)
        total_tiles = max(1, num_tiles_y * num_tiles_x)
        processed_tiles = 0

        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                y_start = max(0, y - tile_pad)
                x_start = max(0, x - tile_pad)
                y_end = min(h, y + tile_size + tile_pad)
                x_end = min(w, x + tile_size + tile_pad)
                
                # Extract padded input tile
                in_tile = img_tensor[:, :, y_start:y_end, x_start:x_end]
                
                with torch.no_grad():
                    if self.device.type == "cuda":
                        with torch.autocast(device_type="cuda"): # Speedup with Mixed Precision
                            out_tile = model(in_tile)
                    else:
                        out_tile = model(in_tile)
                
                # Offset mapping for pasting valid data back
                tile_y_offset = (y - y_start) * scale
                tile_x_offset = (x - x_start) * scale
                
                valid_h = min(tile_size, h - y) * scale
                valid_w = min(tile_size, w - x) * scale
                
                # Crop away the padding
                valid_out_tile = out_tile[:, :, 
                                          tile_y_offset : tile_y_offset + valid_h, 
                                          tile_x_offset : tile_x_offset + valid_w]
                
                # Merge into final output canvas (moves from GPU to CPU here)
                out_y, out_x = y * scale, x * scale
                out_tensor[:, :, out_y : out_y + valid_h, out_x : out_x + valid_w] = valid_out_tile.cpu()
                
                processed_tiles += 1
                if progress_callback:
                    progress_callback(processed_tiles / total_tiles)
                
        return out_tensor

    def _upscale_ai(self, path, progress_callback=None):
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
        
        out_tensor = self._process_with_tiling(img_tensor, model, scale=scale, progress_callback=progress_callback)
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

    def upscale(self, input_path: str, model_type: str, scale_val: str, progress_callback=None) -> Image.Image:
        if not Path(input_path).exists():
            raise ValueError("Input file not found.")

        is_4k = (scale_val == "4K (Ultra HD)")

        img = Image.open(input_path)
        w, h = img.size
        original_mode = img.mode

        # Determine AI constraints based on user choice
        if is_4k:
            target_w, target_h = 3840, 2160
            ratio = min(target_w / w, target_h / h)
            ai_scale = 1 if ratio <= 1.0 else ratio
        else:
            ai_scale = int(scale_val.replace("x", ""))

        # Execution Phase
        if ai_scale <= 1:
            if progress_callback: progress_callback(1.0)
            result = img
        elif self.has_ai and "RealESRGAN" in model_type:
            # AI outputs native 4x 
            # We close the initial img here to save memory since _upscale_ai opens its own
            img.close()
            result = self._upscale_ai(input_path, progress_callback=progress_callback)
            
            # Supersampling: If user selected 2x/3x, we scale the AI 4x result BACK down for sharp quality
            if not is_4k and ai_scale < 4:
                result = result.resize((int(w * ai_scale), int(h * ai_scale)), Image.Resampling.LANCZOS)
        else:
            img.close()
            if progress_callback: progress_callback(0.5)
            result = self._upscale_pil(input_path, math.ceil(ai_scale))
            if progress_callback: progress_callback(1.0)

        # Precisely fit to 4K Bounds if requested
        if is_4k and ratio > 1.0:
            result = self._resize_to_target(result, 3840, 2160)

        return result

        # Precisely fit to 4K Bounds if requested
        if is_4k and ratio > 1.0:
            result = self._resize_to_target(result, 3840, 2160)

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
        
        self.opt_scale = ctk.CTkOptionMenu(self.sidebar, values=["2x", "3x", "4x", "4K (Ultra HD)"])
        self.opt_scale.pack(padx=20, pady=10, fill="x")

        self.btn_run = ctk.CTkButton(self.sidebar, text="Start Upscale", command=self.run_upscale, 
                                     fg_color="#1f6aa5", hover_color="#144870", height=45, state="disabled")
        self.btn_run.pack(padx=20, pady=(40, 10), fill="x")

        self.btn_save = ctk.CTkButton(self.sidebar, text="Save Result", command=self.save_image, state="disabled", height=40)
        self.btn_save.pack(padx=20, pady=10, fill="x")

        # Theme toggle
        self.lbl_theme = ctk.CTkLabel(self.sidebar, text="Theme:", anchor="w")
        self.lbl_theme.pack(padx=20, pady=(20, 0), fill="x", side="bottom")
        self.opt_theme = ctk.CTkOptionMenu(self.sidebar, values=["Dark", "Light"], command=lambda m: ctk.set_appearance_mode(m))
        self.opt_theme.pack(padx=20, pady=(5, 30), fill="x", side="bottom")

        # System Monitor
        self.lbl_cpu = ctk.CTkLabel(self.sidebar, text="System Load: 0%", anchor="w", font=ctk.CTkFont(size=12))
        self.lbl_cpu.pack(padx=20, pady=(10, 0), fill="x", side="bottom")
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
        path = event.data.strip('{}') # Remove curly braces for paths with spaces
        if path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
            self.load_image(path)
        else:
            messagebox.showwarning("Invalid File", "Please drop a valid image file (PNG, JPG, WEBP, BMP).")

    def load_image(self, path):
        self.input_path = path
        self.btn_run.configure(state="normal")
        self.btn_save.configure(state="disabled")
        self.show_preview(path)

    def show_preview(self, path_or_img):
        if isinstance(path_or_img, str):
            img = Image.open(path_or_img).convert("RGB")
        else:
            img = path_or_img.copy()
            
        target_w, target_h = 750, 550
        img.thumbnail((target_w, target_h))
        
        self.preview_tk = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.lbl_preview.configure(image=self.preview_tk, text="")

    def run_upscale(self):
        if not self.input_path: return
        
        model = self.opt_model.get()
        scale_val = self.opt_scale.get() 
        
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
        
        self.progress.set(0)
        self.lbl_status.configure(text="Initializing AI Engine...")
        
        def progress_callback(pct):
            # Use self.after to update UI from the background thread safely
            self.after(0, lambda: self._update_progress_ui(pct))

        threading.Thread(target=self._process, args=(model, scale_val, progress_callback), daemon=True).start()

    def _update_progress_ui(self, pct):
        self.progress.set(pct)
        self.lbl_status.configure(text=f"Processing: {int(pct * 100)}%")

    def _process(self, model, scale_val, progress_callback):
        try:
            self.output_image = self.engine.upscale(self.input_path, model, scale_val, progress_callback=progress_callback)
            self.after(0, self._finish)
        except Exception as e:
            # Enhanced error reporting
            err_msg = str(e)
            if "out of memory" in err_msg.lower():
                err_msg = "GPU Out of Memory! Try a smaller scale or check if other apps are using VRAM."
            
            self.after(0, lambda: messagebox.showerror("Upscale Error", f"Failed: {err_msg}"))
            self.after(0, self._reset_ui)

    def _finish(self):
        self.progress.set(1)
        self.lbl_status.configure(text="Complete! (100%)")
        self.show_preview(self.output_image)
        self.btn_run.configure(state="normal")
        self.btn_select.configure(state="normal")
        self.btn_save.configure(state="normal")
        messagebox.showinfo("Success", "Upscaling complete!")

    def _reset_ui(self):
        self.progress.set(0)
        self.lbl_status.configure(text="Ready")
        self.btn_run.configure(state="normal")
        self.btn_select.configure(state="normal")

    def save_image(self):
        if not self.output_image: return
        path = filedialog.asksaveasfilename(defaultextension=".png", 
                                            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
        if path:
            self.output_image.save(path)
            messagebox.showinfo("Saved", f"Image saved successfully to:\n{path}")

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