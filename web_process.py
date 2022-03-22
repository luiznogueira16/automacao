from distutils.filelist import findall
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys
import os
import requests

url = "https://cndt-certidao.tst.jus.br/inicio.faces"
url_captcha = '//*[@id="corpo"]/div/div[2]/input[1]'

navegador = webdriver.Chrome()
navegador.get(url)
#navegador.get("https://www.google.com/")
#elem = navegador.find_element_by_name("gerarCertidaoForm:cpfCnpj")
#elem.send_keys("2442194172416")

soup = BeautifulSoup(navegador.page_source, "html.parser")

navegador.find_element_by_xpath(url_captcha).click()

try:
    img = navegador.find_element_by_xpath('//*[@id="idImgBase64"]')
    print(img)
    img.screenshot('Download-Location' + "captcha" + ".png")
    
except:
    print("nao deu")