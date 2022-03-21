from selenium import webdriver
from selenium.webdriver.common.keys import Keys

navegador = webdriver.Chrome()
navegador.get("https://cndt-certidao.tst.jus.br/inicio.faces")
#navegador.get("https://www.google.com/")
navegador.find_element_by_xpath('//*[@id="corpo"]/div/div[2]/input[1]').click()
elem = navegador.find_element_by_name("gerarCertidaoForm:cpfCnpj")
elem.send_keys("2442194172416")

