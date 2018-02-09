import numpy as np
import cv2
import os

'''
    image resize module
'''

folders = ['/home/henrique/Área de Trabalho/Projetos/MiojoEdd/img/train/ed',
'/home/henrique/Área de Trabalho/Projetos/MiojoEdd/img/train/miojo',
'/home/henrique/Área de Trabalho/Projetos/MiojoEdd/img/validation/ed',
'/home/henrique/Área de Trabalho/Projetos/MiojoEdd/img/validation/miojo']

def preprocess(path):
  #resize 150x150
  for fileName in os.listdir(path):

    image = cv2.imread(path + "/" + fileName, 0)
 	#force grayscale reading, twice as fast as converting it later   
    if check_img_size(image):
        #true
       	pass
    
    else:
        image = cv2.resize(image, (150, 150), interpolation = cv2.INTER_AREA) 
    
    cv2.imwrite(path + "/" + fileName, image)

def check_img_size(image):

	if image.shape == (150,150,3):
		return True
	
	else:
		return False

if __name__ == "__main__":

    for folder in folders:
    	preprocess(folder)
    
