from soap import segmentation

segmenter = segmentation.Segmenter()

segmenter.load_images_from_directory("data")

segmenter.rescale_images(1000)

segmenter.save_images()

segmenter._parse_to_gray()

segmenter.save_images(image_type="gray")

segmenter.create_masks()

segmenter.save_images(image_type="masks")

# segmenter.create_masks(use_mean=True)

# segmenter.save_images(image_type="masks")
