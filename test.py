from soap import segmentation

segmenter = segmentation.Segmenter()

segmenter.load_images_from_directory("data")

segmenter.rescale_images(500)

segmenter.save_images()

segmenter._parse_to_gray()

segmenter.save_images(gray=True)
