from PIL import Image
import cv2
from cv2 import THRESH_MASK
from cv2 import THRESH_OTSU

metodos = [
    cv2.THRESH_BINARY,
    cv2.THRESH_BINARY_INV,
    cv2.THRESH_TRUNC,
    cv2.THRESH_BINARY,
    cv2.THRESH_TOZERO_INV
]

imagem = cv2.imread("./img/baixados.jpg", 0)

imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BAYER_BG2BGR)

i = 0
for metodo in metodos:
    i += 1
    _, imagem_tratada = cv2.threshold(imagem_cinza, 190, 255, metodo)
    cv2.imwrite(f'teste_metodo/imagem_tratada_{i}.png', imagem_tratada)

imagem = Image.open("teste_metodo/imagem_tratada_3.png")
imagem = imagem.convert("P")
imagem2 = Image.new("P", imagem.size, 255)

for x in range(imagem.size[1]):
    for y in range(imagem.size[0]):
        cor_pixel = imagem.getpixel((y, x))
        if cor_pixel < 139:
            imagem2.putpixel((y, x), 0)
            
imagem2.save('teste_metodo/imagem_final.png')            
        