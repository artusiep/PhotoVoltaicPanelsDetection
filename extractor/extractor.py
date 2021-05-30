import flirimageextractor
from matplotlib import cm
from exif import Image

file_path = '../data/raw/DJI_4_R(600).JPG'

with open(file_path, 'rb') as image_file:
    my_image = Image(image_file)
    # print(my_image.has_exif)

palettes = [cm.jet, cm.bwr, cm.gist_ncar, cm.plasma, cm.turbo]
flir = flirimageextractor.FlirImageExtractor(palettes=palettes)
flir.process_image(file_path)

flir.save_images()
print("Finished")
