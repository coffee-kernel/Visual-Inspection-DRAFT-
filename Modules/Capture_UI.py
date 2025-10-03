import customtkinter as ctk
import subprocess
import os
from datetime import datetime
import logging
from threading import Thread
from tkinter import messagebox
import sys
import tkinter as tk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CameraApp:
    def __init__(self):
        self.camera_initialized = False

    def initialize_camera(self):
        logger.error("Initializing camera...")
        Thread(target=self._initialize_camera_thread).start()

    def _initialize_camera_thread(self):
        try:
            subprocess.run(["udisksctl", "unmount", "-b", "/dev/bus/usb/001/027"], capture_output=True, text=True)

            result = subprocess.run(["gphoto2", "--auto-detect"], capture_output=True, text=True)
            if "Canon EOS R10" not in result.stdout and "USB PTP Class Camera" not in result.stdout:
                logger.error("Camera not detected")
                messagebox.showwarning("Warning", "Camera device is not detected, Please restart the application.")
                return

            result = subprocess.run(["gphoto2", "--summary"], capture_output=True, text=True)
            if result.returncode != 0:
                messagebox.showwarning("Warning", "Failed to initialize camera, Please restart the application.")
                logger.error(f"Camera initialization failed: {result.stderr}")
                return
            
            logger.info("Camera initialized successfully")
            self.camera_initialized = True
            logger.info("Camera ready!")
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")

    def capture_photo(self):
        if not self.camera_initialized:
            logger.info("Error: Camera not initialized")
            return
        
        logger.info("Capturing photo...")
        Thread(target=self._capture_photo_thread).start()

    def _capture_photo_thread(self):
        try:
            subprocess.run(["udisksctl", "unmount", "-b", "/dev/bus/usb/001/027"], capture_output=True, text=True)

            # Directory where image saved
            output_dir = "/home/nvidia/Main_Folder/Inspected_images/Captured_Images"

            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"photo_{timestamp}.jpg")
            cmd = ["gphoto2", "--capture-image-and-download", f"--filename={output_file}"]
            if "USB PTP Class Camera" in subprocess.run(["gphoto2", "--auto-detect"], capture_output=True, text=True).stdout:
                cmd = ["gphoto2", "--camera", "Canon EOS R10", "--port", "usb:001,027", "--capture-image-and-download", f"--filename={output_file}"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Photo captured and saved to {output_file}")
            else:
                logger.error(f"Capture failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Capture error: {str(e)}")