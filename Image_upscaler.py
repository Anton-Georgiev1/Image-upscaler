import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import sys
import threading
from pathlib import Path
import cv2
import numpy as np

# --- AI Engine Integration (Optional) ---
try:
    import torch
    from super_image import EdsrModel, ImageLoader
    HAS_AI = True
except ImportError:
    HAS_AI = False

class UpscalerEngine:
    """Core logic for image upscaling."""
    
    def __init__(self):
        self.has_ai = HAS_AI
        # Check if OpenCV DNN SuperRes is available
        try:
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            self.has_dnn = True
        except AttributeError:
            self.has_dnn = False

    def upscale(self, input_path: str, model_type: str, scale: int) -> Image.Image:
        """Process the image based on selected model and scale."""
        if not Path(input_path).exists():
            raise ValueError("Input file not found.")

        # Priority 1: PyTorch HD Models (via super-image)
        if self.has_ai and "HD" in model_type:
            return self._upscale_pytorch(input_path, scale)

        # Priority 2: OpenCV DNN (Requires .pb files in a 'models' folder)
        if self.has_dnn and model_type in ["EDSR", "ESPCN"]:
            return self._upscale_dnn(input_path, model_type, scale)

        # Default/Fallback: High-Quality PIL Lanczos
        return self._upscale_pil(input_path, scale)

    def _upscale_pytorch(self, path, scale):
        try:
            image = Image.open(path)
            model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=scale)
            inputs = ImageLoader.load_image(image)
            preds = model(inputs)
            # Convert tensor back to PIL
            from torchvision.transforms import ToPILImage
            return ToPILImage()(preds.cpu().detach().squeeze(0))
        except Exception as e:
            print(f"PyTorch failed: {e}")
            return self._upscale_pil(path, scale)

    def _upscale_dnn(self, path, model_name, scale):
        try:
            img = cv2.imread(path)
            model_path = f"models/{model_name}_x{scale}.pb"
            if not os.path.exists(model_path):
                return self._upscale_pil(path, scale)
            
            self.sr.readModel(model_path)
            self.sr.setModel(model_name.lower(), scale)
            result = self.sr.upsample(img)
            return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"DNN failed: {e}")
            return self._upscale_pil(path, scale)

    def _upscale_pil(self, path, scale):
        img = Image.open(path)
        new_size = (img.width * scale, img.height * scale)
        return img.resize(new_size, Image.Resampling.LANCZOS)

# --- GUI Application ---
class ImageUpscalerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Image Upscaler")
        self.geometry("1100x750")
        
        # Initialize Engine
        self.engine = UpscalerEngine()
        
        # State
        self.input_path = None
        self.output_image = None

        # --- Layout ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.lbl_title = ctk.CTkLabel(self.sidebar, text="Image Upscaler", font=ctk.CTkFont(size=22, weight="bold"))
        self.lbl_title.pack(pady=(30, 20))

        self.btn_select = ctk.CTkButton(self.sidebar, text="Open Image", command=self.select_image, height=40)
        self.btn_select.pack(padx=20, pady=10, fill="x")

        self.lbl_model = ctk.CTkLabel(self.sidebar, text="Upscale Model:", anchor="w")
        self.lbl_model.pack(padx=20, pady=(20, 0), fill="x")
        
        model_options = ["EDSR (HD)", "Lanczos (Standard)"]
        if self.engine.has_dnn:
            model_options.extend(["EDSR", "ESPCN"])
            
        self.opt_model = ctk.CTkOptionMenu(self.sidebar, values=model_options)
        self.opt_model.pack(padx=20, pady=10, fill="x")

        self.lbl_scale = ctk.CTkLabel(self.sidebar, text="Scale Factor:", anchor="w")
        self.lbl_scale.pack(padx=20, pady=(10, 0), fill="x")
        self.opt_scale = ctk.CTkOptionMenu(self.sidebar, values=["2", "3", "4"])
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
            img = Image.open(path_or_img)
        else:
            img = path_or_img
            
        # Responsive preview size
        target_w, target_h = 750, 550
        img.thumbnail((target_w, target_h))
        
        self.preview_tk = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.lbl_preview.configure(image=self.preview_tk, text="")

    def run_upscale(self):
        if not self.input_path: return
        
        self.btn_run.configure(state="disabled")
        self.btn_select.configure(state="disabled")
        self.progress.start()
        
        model = self.opt_model.get()
        scale = int(self.opt_scale.get())
        
        threading.Thread(target=self._process, args=(model, scale), daemon=True).start()

    def _process(self, model, scale):
        try:
            self.output_image = self.engine.upscale(self.input_path, model, scale)
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
