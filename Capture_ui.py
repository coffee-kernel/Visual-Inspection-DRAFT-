import customtkinter as ctk
import subprocess
import os
import time
from datetime import datetime
import logging
from threading import Thread

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Capture Controller")
        self.root.geometry("550x180")
        self.center_window()
        self.label_status = ctk.CTkLabel(self.root, text="Connect camera and click 'Initialize'", font=("Arial", 14))
        self.label_status.pack(pady=20)
        self.btn_init = ctk.CTkButton(self.root, text="Initialize Camera", command=self.initialize_camera, font=("Arial", 12))
        self.btn_init.pack(pady=10)
        self.btn_capture = ctk.CTkButton(self.root, text="Capture Photo", command=self.capture_photo, font=("Arial", 12), state="disabled")
        self.btn_capture.pack(pady=10)
        self.camera_initialized = False

    def center_window(self):
        self.root.update_idletasks()
        width = 550
        height = 180
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def initialize_camera(self):
        self.label_status.configure(text="Initializing camera...")
        self.btn_init.configure(state="disabled")
        Thread(target=self._initialize_camera_thread).start()

    def _initialize_camera_thread(self):
        try:
            subprocess.run(["udisksctl", "unmount", "-b", "/dev/bus/usb/001/027"], capture_output=True, text=True)

            result = subprocess.run(["gphoto2", "--auto-detect"], capture_output=True, text=True)
            if "Canon EOS R10" not in result.stdout and "USB PTP Class Camera" not in result.stdout:
                logger.error("Camera not detected")
                self.root.after(0, lambda: self.label_status.configure(text="Error: Camera not detected. Check USB/PTP mode."))
                self.root.after(0, lambda: self.btn_init.configure(state="normal"))
                return
            result = subprocess.run(["gphoto2", "--summary"], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Camera initialization failed: {result.stderr}")
                self.root.after(0, lambda: self.label_status.configure(text="Error: Camera initialization failed."))
                self.root.after(0, lambda: self.btn_init.configure(state="normal"))
                return
            logger.info("Camera initialized successfully")
            self.camera_initialized = True
            self.root.after(0, lambda: self.label_status.configure(text="Camera ready!"))
            self.root.after(0, lambda: self.btn_capture.configure(state="normal"))
            self.root.after(0, lambda: self.btn_init.configure(state="normal"))
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            self.root.after(0, lambda: self.label_status.configure(text=f"Error: {str(e)}"))
            self.root.after(0, lambda: self.btn_init.configure(state="normal"))

    def capture_photo(self):
        if not self.camera_initialized:
            self.label_status.configure(text="Error: Camera not initialized")
            return
        self.label_status.configure(text="Capturing photo...")
        self.btn_capture.configure(state="disabled")
        Thread(target=self._capture_photo_thread).start()

    def _capture_photo_thread(self):
        try:
            subprocess.run(["udisksctl", "unmount", "-b", "/dev/bus/usb/001/027"], capture_output=True, text=True)

            output_dir = "/home/nvidia/Main Folder/Capture_Images"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"photo_{timestamp}.JPG")
            cmd = ["gphoto2", "--capture-image-and-download", f"--filename={output_file}"]
            if "USB PTP Class Camera" in subprocess.run(["gphoto2", "--auto-detect"], capture_output=True, text=True).stdout:
                cmd = ["gphoto2", "--camera", "Canon EOS R10", "--port", "usb:001,027", "--capture-image-and-download", f"--filename={output_file}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Photo captured and saved to {output_file}")
                self.root.after(0, lambda: self.label_status.configure(text=f"Photo saved to {output_file}"))
            else:
                logger.error(f"Capture failed: {result.stderr}")
                self.root.after(0, lambda: self.label_status.configure(text="Error: Capture failed. Check logs."))
            self.root.after(0, lambda: self.btn_capture.configure(state="normal"))
        except Exception as e:
            logger.error(f"Capture error: {str(e)}")
            self.root.after(0, lambda: self.label_status.configure(text=f"Error: {str(e)}"))
            self.root.after(0, lambda: self.btn_capture.configure(state="normal"))

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = ctk.CTk()
    camera_app = CameraApp(app)
    app.mainloop()