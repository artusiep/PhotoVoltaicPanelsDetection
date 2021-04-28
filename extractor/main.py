import flirimageextractor
from matplotlib import cm
from exif import Image

file_path = '../data/raw/sample.JPG'

with open(file_path, 'rb') as image_file:
    my_image = Image(image_file)
    print(my_image.has_exif)

flir = flirimageextractor.FlirImageExtractor(palettes=[cm.jet, cm.bwr, cm.gist_ncar])
flir.process_image(file_path)

flir.save_images()
flir.plot()