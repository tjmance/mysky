#!/usr/bin/env python3
"""
Create a simple icon for MySky application
"""

import os
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Pillow library is required. Install with: pip install Pillow")
    exit(1)


def create_icon():
    """Create a simple icon for the application"""
    # Create a 256x256 image with a gradient background
    size = 256
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw gradient background
    for y in range(size):
        # Blue gradient from light to dark
        r = int(100 + (y / size) * 50)
        g = int(150 + (y / size) * 50)
        b = int(200 + (y / size) * 55)
        draw.line([(0, y), (size, y)], fill=(r, g, b, 255))
    
    # Draw a circle
    margin = 30
    draw.ellipse([margin, margin, size-margin, size-margin], 
                 fill=(255, 255, 255, 200), outline=(50, 50, 50, 255), width=3)
    
    # Draw text "MS" for MySky
    try:
        # Try to use a built-in font
        font = ImageFont.truetype("arial.ttf", 80)
    except:
        # Use default font if arial is not available
        font = ImageFont.load_default()
    
    # Draw text
    text = "MS"
    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    x = (size - text_width) // 2
    y = (size - text_height) // 2 - 10
    
    draw.text((x, y), text, fill=(50, 50, 50, 255), font=font)
    
    # Save as ICO file with multiple sizes
    img.save('mysky.ico', format='ICO', sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
    print("Icon created: mysky.ico")


if __name__ == "__main__":
    create_icon()