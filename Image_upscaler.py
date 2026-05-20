import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import os
import threading
import math
from pathlib import Path
import cv2
import numpy as np
import urllib.request
import sys

# --- Fix for 'basicsr' compatibility with newer torchvision versions ---
try:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
except ImportError:
    pass

# --- AI Engine Integration (Real-ESRGAN) ---
try:
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    HAS_AI = True
except ImportError:
    HAS_AI = False


class UpscalerEngine:
    """Core logic for Real-ESRGAN upscaling & precise 4K scaling."""
    
    def __init__(self):
        self.has_ai = HAS_AI
        self.realesrgan_model = None

    def _init_realesrgan(self):
        """Initializes and caches the RealESRGAN model."""
        if self.realesrgan_model is not None:
            return self.realesrgan_model
            
        weight_path = "models/RealESRGAN_x4plus.pth"
        if not os.path.exists(weight_path):
            print("Downloading RealESRGAN weights (~60MB)...")
            os.makedirs("models", exist_ok=True)
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            urllib.request.urlretrieve(url, weight_path)
            print("Download complete.")
            
        # The x4plus model is an RRDBNet structure scaled by 4x
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        self.realesrgan_model = RealESRGANer(
            scale=4,
            model_path=weight_path,
            model=model,
            tile=256,          # Important: Tiling prevents VRAM crashes on 4K resolutions
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available() # Uses FP16 for speed if Nvidia GPU is available
        )
        return self.realesrgan_model

    def upscale(self, input_path: str, model_type: str, scale_val: str) -> Image.Image:
        """Process the image based on requested scale string and 4K bounds."""
        if not Path(input_path).exists():
            raise ValueError("Input file not found.")

        is_4k = (scale_val == "4K (Ultra HD)")

        # Determine original dimensions
        with Image.open(input_path) as img:
            w, h = img.size
            img = img.convert("RGB")

        # Calculate Needed AI Scale Multiplier
        if is_4k:
            target_w, target_h = 3840, 2160
            ratio = min(target_w / w, target_h / h)
            
            if ratio <= 1.0:
                ai_scale = 1  # Already 4K or bigger
            else:
                ai_scale = math.ceil(ratio)
                if ai_scale > 4: 
                    ai_scale = 4  # Cap AI upscale at 4x to avoid crashing
        else:
            ai_scale = int(scale_val.replace("x", ""))

        # Execute Upscale
        if ai_scale == 1:
            result = img
        elif self.has_ai and "RealESRGAN" in model_type:
            result = self._upscale_realesrgan(input_path, ai_scale)
        else:
            result = self._upscale_pil(input_path, ai_scale)

        # Final Pass: Shrink to precisely 4K if Ultra HD was requested
        if is_4k:
            result = self._resize_to_target(result, 3840, 2160)

        return result

    def _resize_to_target(self, img: Image.Image, target_w: int, target_h: int) -> Image.Image:
        """Resizes safely to strict 4K limits maintaining original aspect ratio."""
        w, h = img.size
        ratio = min(target_w / w, target_h / h)
        new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
        return img.resize(new_size, Image.Resampling.LANCZOS)

    def _upscale_realesrgan(self, path, scale):
        try:
            upsampler = self._init_realesrgan()
            
            # RealESRGAN expects BGR format from OpenCV
            img_cv2 = cv2.imread(path, cv2.IMREAD_COLOR)
            
            # The 'outscale' parameter handles our dynamic scaling multiplier directly
            output_cv2, _ = upsampler.enhance(img_cv2, outscale=scale)
            
            # Convert back to Pillow RGB
            output_cv2 = cv2.cvtColor(output_cv2, cv2.COLOR_BGR2RGB)
            return Image.fromarray(output_cv2)
        except Exception as e:
            print(f"RealESRGAN error: {e}")
            return self._upscale_pil(path, scale)

    def _upscale_pil(self, path, scale):
        """Fallback CPU scaling using Lanczos."""
        img = Image.open(path).convert("RGB")
        new_size = (int(img.width * scale), int(img.height * scale))
        return img.resize(new_size, Image.Resampling.LANCZOS)


# --- GUI Application ---
class ImageUpscalerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Real-ESRGAN Image Upscaler")
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
        
        self.lbl_title = ctk.CTkLabel(self.sidebar, text="AI Upscaler", font=ctk.CTkFont(size=22, weight="bold"))
        self.lbl_title.pack(pady=(30, 20))

        self.btn_select = ctk.CTkButton(self.sidebar, text="Open Image", command=self.select_image, height=40)
        self.btn_select.pack(padx=20, pady=10, fill="x")

        self.lbl_model = ctk.CTkLabel(self.sidebar, text="Upscale Model:", anchor="w")
        self.lbl_model.pack(padx=20, pady=(20, 0), fill="x")
        
        model_options = ["Lanczos (Fast CPU)"]
        if self.engine.has_ai:
            model_options.insert(0, "RealESRGAN (AI HD)")
            
        self.opt_model = ctk.CTkOptionMenu(self.sidebar, values=model_options)
        self.opt_model.pack(padx=20, pady=10, fill="x")

        self.lbl_scale = ctk.CTkLabel(self.sidebar, text="Scale Factor:", anchor="w")
        self.lbl_scale.pack(padx=20, pady=(10, 0), fill="x")
        
        # 4K / Ultra HD Integrated
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

        # Main View
        self.main_view = ctk.CTkFrame(self, corner_radius=15)
        self.main_view.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_view.grid_rowconfigure(0, weight=1)
        self.main_view.grid_columnconfigure(0, weight=1)

        self.lbl_preview = ctk.CTkLabel(self.main_view, text="Select an image to preview", text_color="gray")
        self.lbl_preview.grid(row=0, column=0, sticky="nsew")

        self.progress = ctk.CTkProgressBar(self.main_view)
        self.progress.grid(row=1, column=0, padx=50, pady=(0, 30), sticky="ew")
        self.progress.set(0)

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")])
        if path:
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
        
        # User feedback if downloading the ~60mb weights for the very first time
        if "RealESRGAN" in model and not os.path.exists("models/RealESRGAN_x4plus.pth") and self.engine.has_ai:
            self.lbl_preview.configure(text="Downloading AI Weights (~60MB). Please wait...", image="")
            self.update() 

        self.btn_run.configure(state="disabled")
        self.btn_select.configure(state="disabled")
        self.progress.start()
        
        threading.Thread(target=self._process, args=(model, scale_val), daemon=True).start()

    def _process(self, model, scale_val):
        try:
            self.output_image = self.engine.upscale(self.input_path, model, scale_val)
            self.after(0, self._finish)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Upscale failed: {e}"))
            self.after(0, self._reset_ui)

    def _finish(self):
        self.progress.stop()
        self.progress.set(1)
        self.show_preview(self.output_image)
        self.btn_run.configure(state="normal")
        self.btn_select.configure(state="normal")
        self.btn_save.configure(state="normal")
        messagebox.showinfo("Success", "Upscaling complete!")

    def _reset_ui(self):
        self.progress.stop()
        self.progress.set(0)
        self.btn_run.configure(state="normal")
        self.btn_select.configure(state="normal")

    def save_image(self):
        if not self.output_image: return
        path = filedialog.asksaveasfilename(defaultextension=".png", 
                                            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
        if path:
            self.output_image.save(path)
            messagebox.showinfo("Saved", f"Image saved successfully to:\n{path}")

if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    app = ImageUpscalerApp()
    app.mainloop()