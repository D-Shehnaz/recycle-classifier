# generate_synthetic.py
import os
from PIL import Image, ImageDraw
import random

BASE_DIR = "dataset"
SPLITS = ["train", "val"]
CLASSES = ["recyclable", "non_recyclable"]
IMG_SIZE = (180, 180)

def make_dirs():
    for sp in SPLITS:
        for cl in CLASSES:
            os.makedirs(os.path.join(BASE_DIR, sp, cl), exist_ok=True)

def draw_recyclable(draw):
    # recyclable: blue circle (like bottle), rectangles
    w,h = IMG_SIZE
    r = random.randint(20, 60)
    x = random.randint(r, w-r)
    y = random.randint(r, h-r)
    draw.ellipse((x-r,y-r,x+r,y+r), fill=(30,144,255))  # blue

def draw_non_recyclable(draw):
    # non recyclable: red blob, jagged polygon
    w,h = IMG_SIZE
    points = [(random.randint(10,w-10), random.randint(10,h-10)) for _ in range(6)]
    draw.polygon(points, fill=(220,20,60))

def create_image(kind):
    img = Image.new("RGB", IMG_SIZE, (255,255,255))
    draw = ImageDraw.Draw(img)
    if kind == "recyclable":
        draw_recyclable(draw)
    else:
        draw_non_recyclable(draw)
    return img

def populate():
    for split in SPLITS:
        n = 200 if split=="train" else 50
        for cl in CLASSES:
            folder = os.path.join(BASE_DIR, split, cl)
            for i in range(n):
                img = create_image(cl)
                img.save(os.path.join(folder, f"{cl}_{i}.jpg"))

if __name__ == "__main__":
    make_dirs()
    populate()
    print("Synthetic dataset created under 'dataset/' with train/val and two classes.")
