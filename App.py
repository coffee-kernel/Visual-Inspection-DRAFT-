from typing import Tuple
import cv2
import time
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np

# Modules
from Modules.Annotate import Annotator

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

class SharpeyeApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("SHARPEYE - VC")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width}x{screen_height}+0+0")
        
        # navbar frame
        self.navbar = ctk.CTkFrame(self, height=50)
        self.navbar.pack(side="top", fill="x", padx=10, pady=5)

        # serial number label and entry
        self.serial_label = ctk.CTkLabel(self.navbar, text="Serial Number:", font=("Arial", 14))
        self.serial_label.pack(side="left", padx=10)
        self.serial_entry = ctk.CTkEntry(self.navbar, width=280, placeholder_text="Enter serial number")
        self.serial_entry.pack(side="left", padx=10)

        # button for annotation for trial
        self.annotate_btn = ctk.CTkButton(self.navbar, text="Annotate", width=100, font=("Arial", 14), command=self.run_module_action)
        self.annotate_btn.pack(side="left", padx=5)

        # focus the cursor to serial entry
        self.after(3000, self.serial_entry.focus_set)

        # video display
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        self.annotator = Annotator()

        # GStreamer pipeline
        self.DEVICE_PATH = '/dev/video0'
        self.pipeline = (
            f"v4l2src device={self.DEVICE_PATH} ! "
            "image/jpeg,width=1920,height=1080,framerate=30/1 ! "
            "jpegdec ! video/x-raw ! "  # CPU-based JPEG decoding
            "nvvidconv ! video/x-raw,format=BGRx ! "  # GPU-based format conversion
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink"
        )

        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            error_msg = (f"Error: Could not open device {self.DEVICE_PATH}. "
                         "Device may be busy or pipeline is incorrect.\n"
                         "Test: gst-launch-1.0 v4l2src device=/dev/video0 ! "
                         "image/jpeg,width=1920,height=1080,framerate=30/1 ! "
                         "jpegdec ! nvvidconv ! video/x-raw,format=BGRx ! "
                         "videoconvert ! autovideosink\n"
                         "Check: lsof /dev/video0 and kill any processes using it.")
            self.video_label.configure(text=error_msg, font=("Arial", 12))
            print(error_msg)
            return

        print("Press q to QUIT.")
        self.running = True
        self.screen_width = screen_width
        self.screen_height = screen_height - 80
        self.current_frame = None
        self.update_video()


    def run_module_action(self):
        if self.current_frame is None:
            self.video_label.configure(text="No frame available for annotation", font=("Arial", 12))
            print("No frame available")
            return
        try:
            annotated_frame, result = self.annotator.annotate_frame(self.current_frame)
            self.current_frame = annotated_frame  # Update current frame with annotations
            self.video_label.configure(text=f"Module Result: {result}", font=("Arial", 12))
            # Trigger immediate display update
            self.update_video()
        except Exception as e:
            error_msg = f"Module Error: {str(e)}"
            self.video_label.configure(text=error_msg, font=("Arial", 12))
            print(error_msg)
        

    def update_video(self):
        if not self.running:
            return
        
        start_time = time.time()
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            self.running = False
            self.cap.release()
            return
        
        # Store current frame
        self.current_frame = frame
        
        # Display FPS on frame
        # fps = 1.0 / (time.time() - start_time)
        # cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # resize frame based on fit screen
        frame_height, frame_width = frame.shape[:2]
        scale = min(self.screen_width / frame_width, self.screen_height / frame_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        frame = cv2.resize(frame, (new_width, new_height))

        # convert to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=pil_image)

        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo

        self.after(33, self.update_video)

    def destroy(self):
        self.running = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().destroy()

if __name__ == "__main__":
    app = SharpeyeApp()
    app.protocol("WM_DELETE_WINDOW", app.destroy)
    app.bind('<q>', lambda event: app.destroy())
    app.mainloop()