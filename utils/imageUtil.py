#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2024/8/26 
# @Author  : xionglei
# @Comment :

import os
from PIL import Image, ImageFilter

def expand_and_blur_image(image_path, block_size, blur_radius):
    """
    Expand an image by duplicating its edge pixels in 8 directions and then apply heavy Gaussian blur to the edges.

    :param image_path: Path to the input image.
    :param block_size: Size of the pixel block to duplicate.
    :param blur_radius: Radius for the Gaussian blur.
    :return: Expanded and blurred image.
    """
    # Load the image
    img = Image.open(image_path)
    width, height = img.size

    # Create a new image with extra space for the duplicated blocks
    new_width = width + 2 * block_size
    new_height = height + 2 * block_size
    new_img = Image.new(img.mode, (new_width, new_height))

    # Paste the original image in the center
    new_img.paste(img, (block_size, block_size))

    # Duplicate top and bottom blocks
    for i in range(block_size):
        new_img.paste(img.crop((0, 0, width, block_size)), (block_size, i))
        new_img.paste(img.crop((0, height - block_size, width, height)), (block_size, height + block_size + i))

    # Duplicate left and right blocks
    for i in range(block_size):
        new_img.paste(img.crop((0, 0, block_size, height)), (i, block_size))
        new_img.paste(img.crop((width - block_size, 0, width, height)), (width + block_size + i, block_size))

    # Duplicate corner blocks
    for i in range(block_size):
        for j in range(block_size):
            new_img.putpixel((i, j), img.getpixel((0, 0)))
            new_img.putpixel((new_width - 1 - i, j), img.getpixel((width - 1, 0)))
            new_img.putpixel((i, new_height - 1 - j), img.getpixel((0, height - 1)))
            new_img.putpixel((new_width - 1 - i, new_height - 1 - j), img.getpixel((width - 1, height - 1)))

    # Apply Gaussian blur to the edges
    # Create a mask for the edges
    edge_mask = Image.new("L", (new_width, new_height), 0)
    for i in range(block_size):
        for j in range(new_height):
            edge_mask.putpixel((i, j), 255)
            edge_mask.putpixel((new_width - 1 - i, j), 255)
    for i in range(block_size):
        for j in range(new_width):
            edge_mask.putpixel((j, i), 255)
            edge_mask.putpixel((j, new_height - 1 - i), 255)

    # Apply blur
    blurred_edges = new_img.filter(ImageFilter.GaussianBlur(blur_radius))
    new_img.paste(blurred_edges, mask=edge_mask)

    return new_img

def save_image(image_path, output_path):
    """
    保存图片
    """
    expanded_and_blurred_image = expand_and_blur_image(image_path, 500, 100)
    expanded_and_blurred_image.save(os.path.join(output_path))


if __name__ == '__main__':
    save_image('./images/tiled_cropped_image.png', './images/blurred_tiled_cropped_image.png')