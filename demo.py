"""pokedex style demo app. hit the yellow button (or enter) to pick a pic and it shows the top 5 guesses

run: .venv/Scripts/python demo.py   (or just the run button in vs code)
"""
import subprocess
import sys
from pathlib import Path

# if something other than the .venv python started this (like the vs code run button) restart with the .venv one since that's where torch is
VENV_PY = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
if VENV_PY.exists() and Path(sys.executable).resolve() != VENV_PY.resolve():
    sys.exit(subprocess.call([str(VENV_PY), __file__]))

import tkinter as tk
from tkinter import filedialog

import torch
from PIL import Image, ImageTk

from predict import load_trained, predict_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, classes = load_trained("refine/cosine30_s42", device)

RED, RED_DARK, EDGE = "#dc1c1c", "#a00000", "#5a0000"
LCD_INK = "#40ff60"
SCREEN_BLUE = (168, 200, 224)
PROMPT = "PRESS THE YELLOW\nBUTTON TO SCAN"


def to_pixels(path) -> Image.Image:
    """pixelated 32 color version for the screen, transparent parts show the screen blue"""
    with Image.open(path) as im:
        rgba = im.convert("RGBA")
    img = Image.alpha_composite(Image.new("RGBA", rgba.size, SCREEN_BLUE + (255,)), rgba).convert("RGB")
    img.thumbnail((56, 56))
    return img.quantize(32).convert("RGB").resize((img.width * 2, img.height * 2), Image.NEAREST)


root = tk.Tk()
root.title("Species Scanner")
root.resizable(False, False)
c = tk.Canvas(root, width=600, height=400, bg="#26262e", highlightthickness=0)
c.pack()

# left side
c.create_polygon(20, 20, 170, 20, 200, 60, 290, 60, 290, 380, 20, 380, fill=RED, outline=EDGE, width=3)
c.create_line(20, 92, 290, 92, fill=RED_DARK, width=2)
c.create_oval(30, 25, 90, 85, fill="white", outline=EDGE, width=2)
c.create_oval(36, 31, 84, 79, fill="#2a8af0", outline="#1a4ea0", width=2)
c.create_oval(46, 40, 60, 52, fill="#bfe0ff", outline="")
for x, color in ((112, "#ff4040"), (134, "#f0d020"), (156, "#30c040")):
    c.create_oval(x - 7, 30, x + 7, 44, fill=color, outline=EDGE)

c.create_polygon(45, 110, 265, 110, 265, 280, 70, 280, 45, 255, fill="#e8e8e8", outline="#555", width=2)
c.create_rectangle(72, 130, 238, 252, fill="#%02x%02x%02x" % SCREEN_BLUE, outline="#444", width=2)
for x in (145, 165):
    c.create_oval(x - 3, 116, x + 3, 122, fill=RED, outline="")
c.create_oval(78, 262, 88, 272, fill=RED, outline=EDGE)
for y in (262, 266, 270):
    c.create_line(222, y, 248, y, fill="#555", width=2)
screen_img = c.create_image(155, 191)

c.create_oval(38, 295, 62, 319, fill="#222", outline="black", tags="clear")
c.create_text(50, 330, text="CLR", fill="white", font=("Segoe UI", 7, "bold"))
c.create_rectangle(80, 304, 115, 310, fill=RED_DARK, outline=EDGE)
c.create_rectangle(122, 304, 157, 310, fill="#3060c0", outline="#1a3a80")
c.create_rectangle(70, 340, 160, 368, fill="#30c040", outline="#105010", width=2)
name_box = c.create_text(115, 354, text="", fill="#0a300a", font=("Consolas", 8, "bold"))
c.create_rectangle(211, 310, 229, 364, fill="#222", outline="")
c.create_rectangle(193, 328, 247, 346, fill="#222", outline="")

c.create_rectangle(290, 40, 310, 380, fill=RED_DARK, outline=EDGE, width=2)
for y in range(60, 380, 40):
    c.create_line(293, y, 307, y, fill=EDGE)

# right side
c.create_rectangle(310, 60, 580, 380, fill=RED, outline=EDGE, width=3)
c.create_rectangle(330, 78, 560, 178, fill="#1a1a1a", outline="black", width=2)
screen_text = c.create_text(445, 128, text=PROMPT, fill=LCD_INK, font=("Consolas", 10, "bold"), justify="center")
for row in range(2):
    for col in range(5):
        x, y = 330 + col * 46, 196 + row * 27
        c.create_rectangle(x, y, x + 44, y + 25, fill="#3a8ef0", outline="#1a4ea0")
for x in (335, 364):
    c.create_rectangle(x, 268, x + 25, 288, fill="white", outline="#888")
c.create_line(430, 278, 452, 278, fill="#333", width=3)
c.create_line(462, 278, 484, 278, fill="#333", width=3)
c.create_oval(516, 264, 544, 292, fill="#f0c020", outline="#a07000", width=2, tags="scan")
c.create_oval(522, 269, 530, 276, fill="#fff2a0", outline="", tags="scan")
c.create_text(530, 302, text="SCAN", fill="white", font=("Segoe UI", 7, "bold"))
c.create_rectangle(335, 325, 425, 355, fill="#333", outline="black")
c.create_rectangle(470, 325, 560, 355, fill="#333", outline="black")


def scan(_=None):
    path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All files", "*.*")])
    if not path:
        return
    photo = ImageTk.PhotoImage(to_pixels(path))
    c.itemconfig(screen_img, image=photo)
    c.photo = photo  # need to keep a reference or tkinter drops the image
    top = predict_image(path, model, classes, device)
    c.itemconfig(name_box, text=top[0][0].upper()[:12])
    c.coords(screen_text, 345, 128)
    c.itemconfig(screen_text, anchor="w", justify="left",
                 text="\n".join(f"{name.upper()[:14]:<14}{p:7.1%}" for name, p in top))


def clear(_=None):
    c.itemconfig(screen_img, image="")
    c.itemconfig(name_box, text="")
    c.coords(screen_text, 445, 128)
    c.itemconfig(screen_text, anchor="center", justify="center", text=PROMPT)


c.tag_bind("scan", "<Button-1>", scan)
c.tag_bind("clear", "<Button-1>", clear)
root.bind("<Return>", scan)
root.bind("<Escape>", clear)
root.mainloop()
