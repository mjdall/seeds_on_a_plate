import os
import pandas as pd
import numpy as np
import dask
import math
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray

class Segmenter():
    def __init__(self, x_width=-1):
        self.images = None
        self.gray_images = None
        pass


    def load_images_from_directory(self, directory):
        files_in_dir = os.listdir(directory)
        filepaths = [os.path.join(directory, file) for file in files_in_dir]
        self.load_images_from_filenames(filepaths)


    def load_images_from_filenames(self, filepath_list):
        if isinstance(filepath_list, str):
            filepath_list = [filepath_list]

        self.images = [
            plt.imread(filepath) for filepath in filepath_list]


    def rescale_images(self, x_width):
        if x_width <= 100:
            raise RuntimeError("x_width should be >= 100")

        print(f"Rescaling images to have an x width of: {x_width}")

        # get the dimensions to scale each image using
        dimensions = [(image.shape[1], image.shape[0]) for image in self.images]

        rescaled_dims = [
            (math.ceil(x_width * (y_dim / x_dim)), x_width)
            for (x_dim, y_dim) in dimensions]

        resize_list = []
        for (image, new_dims) in zip(self.images, rescaled_dims):
            resize_list.append(
                dask.delayed(resize)(image, new_dims, anti_aliasing=True))

        self.images = dask.compute(resize_list)[0]


    def _parse_to_gray(self):
        if self.images is None:
            raise RuntimeError("no images associated with segmenter")

        print("Processing images to gray scale")
        update_list = []
        for image in self.images:
            update_list.append(dask.delayed(rgb2gray)(image))
        
        self.gray_images = dask.compute(update_list)[0]


    def _parse_gray_image(self, gray_image, gray_filter=.5):
        x_positions = []
        y_positions = []
        gray_values = []
        for y in range(gray_image.shape[0]):
            for x in range(gray_image.shape[1]):
                gray_value = gray_image[y][x]
                if gray_value <= gray_filter:
                    x_positions.append(x)
                    y_positions.append(y)
                    gray_values.append(gray_value)
        return(pd.DataFrame({ "x_pos": x_positions, "y_pos": y_positions, "gray_value": gray_values }))


    def _parse_gray_images(self):
        if self.gray is None:
            print("parsing images to gray scale first")
            self._parse_to_gray()
        
        dask_computation = []
        for image in self.gray_images:
            dask_computation.append(dask.delayed(self._parse_gray_image)(image))

        self.gray_pixels = dask.compute(dask_computation)


    def plot_images(self, gray=False):
        images_to_plot = self.gray_images if gray else self.images
        for image in images_to_plot:
            if gray:
                plt.imshow(image, cmap=plt.get_cmap("gray"))
            else:
                plt.imshow(image)

    
    def save_images(self, prefix="image",
                    output_dir="output\\resized", gray=False):
        images_to_save = self.images
        if gray:
            images_to_save = self.gray_images
            if output_dir == "output\\resized":
                output_dir = "output\\gray"

        os.makedirs(output_dir, exist_ok=True)

        saved_images = []
        for i, image in enumerate(images_to_save):
            image_filename = os.path.join(output_dir, f"{prefix}_{i}.jpg")
            cmap = plt.get_cmap("gray") if gray else None
            plt.imsave(image_filename, image, cmap=cmap)

            saved_images.append(image_filename)
            print(f"image saved to: {image_filename}")

        return(saved_images)
