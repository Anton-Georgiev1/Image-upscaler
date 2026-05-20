import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import os
import threading
import math
from pathlib import Path
import urllib.request

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
        """Downloads (if missing) and caches the RealESRGAN model directly."""
        if self.ai_model is not None:
            return self.ai_model
            
        weight_path = "models/RealESRGAN_x4plus.pth"
        if not os.path.exists(weight_path):
            print("Downloading RealESRGAN weights (~60MB)...")
            os.makedirs("models", exist_ok=True)
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            urllib.request.urlretrieve(url, weight_path)
            print("Download complete.")
            
        # Spandrel natively reads the architecture from the .pth file
        model = ModelLoader().load_from_file(weight_path)
        self.ai_model = model.to(self.device).eval()
        return self.ai_model

    def _process_with_tiling(self, img_tensor, model, scale=4, tile_size=256, tile_pad=16):
        """Processes the image in small chunks to prevent VRAM crashes on 4K bounds."""
        b, c, h, w = img_tensor.shape
        out_h, out_w = h * scale, w * scale
        out_tensor = torch.zeros((b, c, out_h, out_w), device=self.device)
        
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
                
                # Merge into final output canvas
                out_y, out_x = y * scale, x * scale
                out_tensor[:, :, out_y : out_y + valid_h, out_x : out_x + valid_w] = valid_out_tile
                
        return out_tensor

    def _upscale_ai(self, path):
        """Converts to tensor, runs PyTorch inference, converts back to Image."""
        img = Image.open(path).convert("RGB")
        img_tensor = ToTensor()(img).unsqueeze(0).to(self.device)
        
        model = self._init_ai_model()
        scale = model.scale
        
        out_tensor = self._process_with_tiling(img_tensor, model, scale=scale)
        out_img = ToPILImage()(out_tensor.squeeze(0).cpu().clamp(0, 1))
        
        # Free up memory safely
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            
        return out_img

    def upscale(self, input_path: str, model_type: str, scale_val: str) -> Image.Image:
        if not Path(input_path).exists():
            raise ValueError("Input file not found.")

        is_4k = (scale_val == "4K (Ultra HD)")

        with Image.open(input_path) as img:
            w, h = img.size
            img = img.convert("RGB")

        # Determine AI constraints based on user choice
        if is_4k:
            target_w, target_h = 3840, 2160
            ratio = min(target_w / w, target_h / h)
            ai_scale = 1 if ratio <= 1.0 else ratio
        else:
            ai_scale = int(scale_val.replace("x", ""))

        # Execution Phase
        if ai_scale <= 1:
            result = img
        elif self.has_ai and "RealESRGAN" in model_type:
            # AI outputs native 4x 
            result = self._upscale_ai(input_path)
            
            # Supersampling: If user selected 2x/3x, we scale the AI 4x result BACK down for sharp quality
            if not is_4k and ai_scale < 4:
                result = result.resize((int(w * ai_scale), int(h * ai_scale)), Image.Resampling.LANCZOS)
        else:
            result = self._upscale_pil(input_path, math.ceil(ai_scale))

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
        img = Image.open(path).convert("RGB")
        new_size = (int(img.width * scale), int(img.height * scale))
        return img.resize(new_size, Image.Resampling.LANCZOS)


# --- GUI Application ---
class ImageUpscalerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Real-ESRGAN Upscaler (Spandrel Engine)")
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
        
        # User feedback during initial weights download
        if "RealESRGAN" in model and not os.path.exists("models/RealESRGAN_x4plus.pth"):
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