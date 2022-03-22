from PIL import Image

img = Image.open("img/baixados1.png")
img = img.convert("RGBA")
pixdata = img.load()

for y in xrange(img.size[1]):
    for x in xrange(img.size[0]):
        if pixdata[x,y] != (0,0,0,255):
            pixdata[x,y] = (255, 255, 255, 255)
            
img.save("tratado.png", "PNG")

im_orig = Image.open("tratado.png")
big = im_orig.resize((116, 56), Image.NEAREST)

ext = ".tif"
big.save("input-NEAREST" + ext)

from pytesser import * 

image = Image.open("input-NEAREST.tif")

print(image_to_string(image))