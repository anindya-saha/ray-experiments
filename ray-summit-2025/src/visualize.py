from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
import textwrap
import numpy as np

captioned_dataset_path = './outputs/04_captioned_dataset.parquet'
image_with_captions_path = './outputs/image_with_captions.png'

num_samples = 9

# Load data
table = pq.read_table(captioned_dataset_path)
data = table.to_pandas()
samples = data.sample(num_samples)

# Configuration
COLS = 3
ROWS = (num_samples + COLS - 1) // COLS
IMAGE_SIZE = (300, 300)  # Fixed size for all images
CAPTION_HEIGHT = 0.25  # Height ratio for caption box
PADDING = 0.02

# Create figure with white background
fig = plt.figure(figsize=(COLS * 5, ROWS * 8), facecolor='white')

# Create grid spec for better control
gs = fig.add_gridspec(
    ROWS, COLS, 
    hspace=0.8,  # Vertical spacing
    wspace=0.2,  # Horizontal spacing
    left=0.05, right=0.95,
    top=0.95, bottom=0.05
)

for idx, (_, row) in enumerate(samples.head(num_samples).iterrows()):
    # Create subplot
    ax = fig.add_subplot(gs[idx // COLS, idx % COLS])
    
    # Load and resize image to fixed size
    image = Image.open(BytesIO(row['image']['bytes']))
    # Resize maintaining aspect ratio and padding if needed
    image.thumbnail(IMAGE_SIZE, Image.Resampling.LANCZOS)
    
    # Create a new image with exact size and white background
    fixed_image = Image.new('RGB', IMAGE_SIZE, (255, 255, 255))
    # Paste the resized image in the center
    x = (IMAGE_SIZE[0] - image.width) // 2
    y = (IMAGE_SIZE[1] - image.height) // 2
    fixed_image.paste(image, (x, y))
    
    # Display image
    ax.imshow(fixed_image)
    ax.axis('off')
    
    # Prepare caption text
    caption = row['caption']
    wrapped_caption = textwrap.fill(caption, width=50)
    
    # If caption is too long, truncate with ellipsis
    # if len(wrapped_caption) < len(caption):
    #     lines = wrapped_caption.split('\n')
    #     if len(lines) >= 4:
    #         lines[3] = lines[3][:-3] + '...'
    #         wrapped_caption = '\n'.join(lines[:4])
    
    # Add caption in a fixed-size box below the image
    # Create a text box with consistent positioning
    caption_box = patches.FancyBboxPatch(
        (0.0, -0.45),  # Position below image (moved down)
        1.0, 0.40,     # Width and height of box (increased height)
        boxstyle="round,pad=0.02",
        facecolor='#f0f0f0',
        edgecolor='#cccccc',
        linewidth=1,
        transform=ax.transAxes
    )
    ax.add_patch(caption_box)
    
    # Add text to the box
    ax.text(0.5, -0.25, wrapped_caption,
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=10,
            fontfamily='sans-serif',
            color='#333333',
            multialignment='center')
    
    # Add subtle image border
    rect = patches.Rectangle(
        (0, 0), 1, 1,
        linewidth=2,
        edgecolor='#dddddd',
        facecolor='none',
        transform=ax.transAxes
    )
    ax.add_patch(rect)

# Add title to the figure
fig.suptitle('Image Captioning Results', fontsize=20, fontweight='bold', y=0.98)

# Save with high quality
plt.savefig(image_with_captions_path, dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print(f"Figure saved to: {image_with_captions_path}")
plt.show()