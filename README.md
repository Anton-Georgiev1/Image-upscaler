# Image_upscaler

**Breathe new life into your low-resolution images with AI-powered clarity.**

![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-AI%20Engine-EE4C2C?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

`Image_upscaler` is a modern, desktop application designed to transform grainy, low-res photos into sharp, high-definition masterpieces. Whether you're preserving old family memories or preparing assets for a 4K display, this tool provides a simple, professional interface to harness the power of state-of-the-art AI.

---

## Why Image_upscaler?

We believe that every pixel tells a story. Sometimes those stories are blurred by time or technology. `Image_upscaler` bridges that gap by using **Real-ESRGAN** (via the Spandrel architecture) to intelligently reconstruct missing details, rather than just stretching pixels.

### Key Features:
*   **AI-Powered HD:** Uses Real-ESRGAN x4plus for professional-grade upscaling.
*   **4K Targeting:** Automatically scales and fits images precisely to 3840x2160 Ultra HD bounds while maintaining aspect ratio.
*   **Tiling Technology:** Processes large images in manageable chunks ("tiles") to ensure stability even on modest hardware and prevent VRAM crashes.
*   **Fast Fallback:** Includes high-quality Lanczos (CPU) upscaling for quick drafts or when AI isn't required.
*   **Modern Interface:** A sleek, dark-themed GUI built with CustomTkinter for a seamless user experience.

---

## Getting Started

### Prerequisites
*   **Python 3.12+**
*   **CUDA-capable GPU** (Optional, but highly recommended for speed)

### Installation

1.  **Clone the repository:**
2.  **Install dependencies:**
    ```bash
    pip install customtkinter spandrel torch torchvision pillow numpy tkinterdnd2 psutil
    ```

### Model Setup

To use the AI HD mode, you need the **Real-ESRGAN x4plus** weights:

1.  Download `RealESRGAN_x4plus.pth` (e.g., from the [official Real-ESRGAN repository](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)).
2.  Create a folder named `models` in the project root.
3.  Place the `.pth` file inside:
    `./models/RealESRGAN_x4plus.pth`

---

##  Usage

1.  Launch the application:
    ```bash
    python Image_upscaler.py
    ```
2.  **Open Image:** Select your source file or **Drag & Drop** an image directly onto the preview area.
3.  **Choose Model:** Select "RealESRGAN (AI HD)" for best quality.
4.  **Set Scale:** Choose a multiplier (2x, 3x, 4x) or "4K (Ultra HD)".
5.  **Upscale:** Hit "Start Upscale" and watch the progress bar.
6.  **Save:** Once complete, save your high-res result!

---

## Testing

The project maintains a rigorous testing suite to ensure image processing integrity.

```bash
pytest test_image_upscaler.py
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.


