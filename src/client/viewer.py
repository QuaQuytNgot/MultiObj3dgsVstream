"""Optional Tk viewer. Camera and controls do not depend on the renderer API."""
from __future__ import annotations

import numpy as np


class InteractiveViewer:
    def __init__(self, camera_sample, object_ids, *, up_axis="y"):
        try:
            import tkinter as tk
            from PIL import Image, ImageTk
            self.root = tk.Tk()
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError("Interactive mode needs tkinter, Pillow, and a graphical display") from exc
        except Exception as exc:
            raise RuntimeError(f"Cannot open interactive display: {exc}") from exc
        self.Image, self.ImageTk = Image, ImageTk
        self.camera_sample = camera_sample
        self.up_axis = up_axis
        self.object_ids = tuple(object_ids)
        self.selected = 0
        self.paused = False
        self.closed = False
        self.qualities = {}
        self.toggle_objects = []
        self.root.title("3DGS | arrows: orbit  +/-: zoom  space: pause  tab: object  1–4: Q0–Q3  p: object pause")
        self.label = tk.Label(self.root)
        self.label.pack()
        self.root.bind("<KeyPress>", self._key)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _key(self, event):
        key = event.keysym
        if key == "Escape":
            self.close()
        elif key == "space":
            self.paused = not self.paused
        elif key == "Tab":
            self.selected = (self.selected + 1) % len(self.object_ids)
        elif key in ("1", "2", "3", "4"):
            self.qualities[self.object_ids[self.selected]] = int(key) - 1
        elif key in ("p", "P"):
            self.toggle_objects.append(self.object_ids[self.selected])
        elif key in ("Left", "Right"):
            self.camera_sample["azimuth"] = self.camera_sample.get("azimuth", 0) + (5 if key == "Right" else -5)
        elif key in ("Up", "Down"):
            self.camera_sample["elevation"] = min(89, max(-89, self.camera_sample.get("elevation", 0) + (5 if key == "Up" else -5)))
        elif key in ("plus", "equal", "minus"):
            self.camera_sample["distance"] = max(.05, self.camera_sample.get("distance", 4) * (.9 if key != "minus" else 1.1))
        elif key in ("a", "d", "w", "s"):
            target = list(self.camera_sample.get("target", [0, 0, 0]))
            axis = 0 if key in ("a", "d") else (2 if getattr(self, "up_axis", "y") == "z" else 1)
            target[axis] += .05 if key in ("d", "w") else -.05
            self.camera_sample["target"] = target

    def poll(self):
        if not self.closed:
            self.root.update_idletasks()
            self.root.update()
        return not self.closed

    def show(self, rgb):
        self.photo = self.ImageTk.PhotoImage(self.Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255)))
        self.label.configure(image=self.photo)

    def close(self):
        if not self.closed:
            self.closed = True
            self.root.destroy()
