import tkinter as tk
import subprocess
import cv2
from PIL import Image, ImageTk

class ROS2Gui:
    def __init__(self, root):
        self.root = root
        self.root.title("ROS2ノード管理")
        self.proc = None

        self.start_btn = tk.Button(self.root, text="ノード起動")
        self.start_btn.pack(pady=10)
        self.start_btn.bind('<Button-1>', self.start_node)

        self.stop_btn = tk.Button(self.root, text="ノード停止")
        self.stop_btn.pack(pady=10)
        self.stop_btn.bind('<Button-1>', self.stop_node)

        self.fire_btn = tk.Button(self.root, text="発射")
        self.fire_btn.pack(pady=10)
        self.fire_btn.bind('<Button-1>', self.sample_node)

        self.var = tk.StringVar(self.root)
        self.var.set("Mode Select")
        self.dropdown = tk.OptionMenu(self.root, self.var, "Easy", "Medium", "Hard", "Impossible", "?")
        self.dropdown.pack(pady=10)

        self.cap = cv2.VideoCapture(0)
        self.img_label = tk.Label(self.root)
        self.img_label.pack(pady=10)
        self.update_image()

    def start_node(self, event=None):
        btn = event.widget if event else None
        name = btn.cget("text") if btn else "unknown"
        print(f"Pressed button: {name}")
        if self.proc is None:
            self.proc = subprocess.Popen(["ros2", "run", "pingpong", "image_publisher"])

    def stop_node(self, event=None):
        btn = event.widget if event else None
        name = btn.cget("text") if btn else "unknown"
        print(f"Pressed button: {name}")
        if self.proc is not None:
            self.proc.terminate()
            self.proc = None

    def sample_node(self, event=None):
        btn = event.widget if event else None
        name = btn.cget("text") if btn else "unknown"
        print(f"Pressed button: {name}")

    def update_image(self):
        ret, frame = self.cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.img_label.imgtk = imgtk
            self.img_label.configure(image=imgtk)
        self.root.after(30, self.update_image)

        

if __name__ == "__main__":
    root = tk.Tk()
    app = ROS2Gui(root)
    var = tk.StringVar(root)
    label = tk.Label(root, textvariable=var)
    label.pack()
    root.mainloop()