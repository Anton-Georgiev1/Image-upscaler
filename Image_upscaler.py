import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from pathlib import Path
import threading
from upscaler_engine import UpscalerEngine

class ImageUpscalerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Image Upscaler")
        self.geometry("1000x700")
        
        # Initialize engine
        self.engine = UpscalerEngine()
        
        # State variables
        self.input_path = None
        self.output_image = None
        self.preview_image = None

        # Configure layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar frame
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="AI Upscaler", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Select Image Button
        self.select_button = ctk.CTkButton(self.sidebar_frame, text="Select Image", command=self.select_image)
        self.select_button.grid(row=1, column=0, padx=20, pady=10)

        # Model Selection
        self.model_label = ctk.CTkLabel(self.sidebar_frame, text="Model:", anchor="w")
        self.model_label.grid(row=2, column=0, padx=20, pady=(10, 0))
        self.model_option = ctk.CTkOptionMenu(self.sidebar_frame, values=["EDSR (HD)", "MSRN (HD)", "EDSR", "ESPCN", "FSRCNN", "LapSRN"])
        self.model_option.grid(row=3, column=0, padx=20, pady=10)

        # Scale Selection
        self.scale_label = ctk.CTkLabel(self.sidebar_frame, text="Scale Factor:", anchor="w")
        self.scale_label.grid(row=4, column=0, padx=20, pady=(10, 0))
        self.scale_option = ctk.CTkOptionMenu(self.sidebar_frame, values=["2", "3", "4"])
        self.scale_option.grid(row=5, column=0, padx=20, pady=10)

        # Action Buttons
        self.upscale_button = ctk.CTkButton(self.sidebar_frame, text="Upscale", command=self.start_upscale, state="disabled")
        self.upscale_button.grid(row=7, column=0, padx=20, pady=10)

        self.save_button = ctk.CTkButton(self.sidebar_frame, text="Save Result", command=self.save_image, state="disabled")
        self.save_button.grid(row=8, column=0, padx=20, pady=10)

        # Appearance Mode
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=9, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark"],
                                                                       command=self.change_appearance_mode)
        self.appearance_mode_optionemenu.grid(row=10, column=0, padx=20, pady=(10, 20))

        # Main Preview Frame
        self.preview_frame = ctk.CTkFrame(self)
        self.preview_frame.grid(row=0, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.preview_frame.grid_rowconfigure(0, weight=1)
        self.preview_frame.grid_columnconfigure(0, weight=1)

        self.image_label = ctk.CTkLabel(self.preview_frame, text="No Image Selected")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        self.progress_bar = ctk.CTkProgressBar(self.preview_frame)
        self.progress_bar.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.progress_bar.set(0)

    def change_appearance_mode(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if file_path:
            self.input_path = file_path
            self.show_preview(file_path)
            self.upscale_button.configure(state="normal")
            self.save_button.configure(state="disabled")
            self.output_image = None

    def show_preview(self, path):
        img = Image.open(path)
        # Resize for preview
        img.thumbnail((800, 600))
        self.preview_image = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.preview_image, text="")

    def start_upscale(self):
        if not self.input_path:
            return

        self.upscale_button.configure(state="disabled")
        self.progress_bar.set(0)
        self.progress_bar.start()
        
        model = self.model_option.get().lower()
        scale = int(self.scale_option.get())
        
        # Run upscaling in a thread to keep GUI responsive
        thread = threading.Thread(target=self.run_upscale, args=(model, scale))
        thread.start()

    def run_upscale(self, model, scale):
        try:
            self.output_image = self.engine.upscale_image(self.input_path, model, scale)
            self.after(0, self.upscale_finished)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Upscaling failed: {e}"))
            self.after(0, self.upscale_failed)

    def upscale_finished(self):
        self.progress_bar.stop()
        self.progress_bar.set(1)
        self.save_button.configure(state="normal")
        self.upscale_button.configure(state="normal")
        
        # Show output preview
        preview_copy = self.output_image.copy()
        preview_copy.thumbnail((800, 600))
        self.preview_image = ImageTk.PhotoImage(preview_copy)
        self.image_label.configure(image=self.preview_image)
        
        messagebox.showinfo("Success", "Image upscaled successfully!")

    def upscale_failed(self):
        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.upscale_button.configure(state="normal")

    def save_image(self):
        if self.output_image:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            if file_path:
                self.output_image.save(file_path)
                messagebox.showinfo("Saved", f"Image saved to {file_path}")

if __name__ == "__main__":
    app = ImageUpscalerApp()
    app.mainloop()
