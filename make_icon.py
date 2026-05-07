from PIL import Image, ImageDraw
import math
import os

print(f"Working directory: {os.getcwd()}")

size = 256
img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

cx, cy = size // 2, size // 2

# Black circle background
draw.ellipse([4, 4, 251, 251], fill=(0, 0, 0, 255))

# Outer ring
draw.ellipse([4, 4, 251, 251], outline=(0, 247, 255, 255), width=3)

# Middle ring
draw.ellipse([20, 20, 235, 235], outline=(0, 247, 255, 60), width=1)

# Hex segments around outer ring
for i in range(6):
    angle = math.radians(i * 60)
    x1 = cx + 110 * math.cos(angle)
    y1 = cy + 110 * math.sin(angle)
    x2 = cx + 120 * math.cos(angle)
    y2 = cy + 120 * math.sin(angle)
    draw.line([x1, y1, x2, y2], fill=(0, 247, 255, 255), width=3)

# Eye shape - almond/lens outline
eye_width = 100
eye_height = 45
eye_points = []
steps = 60
for i in range(steps):
    angle = math.radians(i * 360 / steps)
    x = cx + eye_width * math.cos(angle)
    y = cy + eye_height * math.sin(angle) * abs(math.cos(angle * 0.5))
    eye_points.append((x, y))

# Eye white/glow area
draw.ellipse([cx - eye_width + 10, cy - eye_height,
              cx + eye_width - 10, cy + eye_height],
             fill=(0, 20, 30, 255), outline=(0, 247, 255, 255), width=2)

# Iris
draw.ellipse([cx - 28, cy - 28, cx + 28, cy + 28],
             fill=(0, 40, 60, 255), outline=(0, 247, 255, 255), width=2)

# Iris inner ring
draw.ellipse([cx - 18, cy - 18, cx + 18, cy + 18],
             outline=(0, 247, 255, 160), width=1)

# Pupil
draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10],
             fill=(0, 247, 255, 255))

# Pupil center dark
draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5],
             fill=(0, 0, 0, 255))

# Glint
draw.ellipse([cx + 4, cy - 8, cx + 9, cy - 3],
             fill=(255, 255, 255, 200))

# Scanlines
for y in range(0, size, 6):
    draw.line([4, y, 251, y], fill=(0, 247, 255, 10), width=1)

# Corner ticks
ticks = [(10, 10, 30, 10), (10, 10, 10, 30),
         (225, 10, 245, 10), (245, 10, 245, 30),
         (10, 225, 10, 245), (10, 245, 30, 245),
         (245, 225, 245, 245), (225, 245, 245, 245)]
for t in ticks:
    draw.line(t, fill=(0, 247, 255, 180), width=2)

save_path = os.path.join(os.getcwd(), "jarvis_icon.png")
ico_path = os.path.join(os.getcwd(), "jarvis_icon.ico")

img.save(save_path)
print(f"PNG saved to: {save_path}")

img.save(ico_path, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)])
print(f"ICO saved to: {ico_path}")