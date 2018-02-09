import cv2
import os

folder = './img/train/ed/'
folder2 = './img/train/miojo/'

'''
    renameImages()
    args:
        -folder: pasta onde estão armazendas as imagens a serem renomeadas
        -prefix: prefixo para renomear. ex: abc.jpeg , prefixo_1.jpg
'''

def resizeImages(path):
  #resize 150x150
  for fileName in os.listdir(path):
      
    print("resize:" + fileName)
    
    image = cv2.imread(path + "/" + fileName)
    
    resized_image = cv2.resize(image, (150, 150)) 
    
    cv2.imwrite(path + "/" + fileName, resized_image)


def renameImages(folder, prefix):
    for root, dirs, files in os.walk(folder):
        for i,f in enumerate(files):
            
            absname = os.path.join(root, f)
            ending = str(absname[-4:])
            print(absname,ending)
            
            if ending == '.jpg':    
                newname = os.path.join(root, prefix+str(i)+".jpg")
                os.rename(absname, newname)
            else:
                newname = os.path.join(root, prefix+str(i)+".png")
                os.rename(absname, newname)

if __name__ == "__main__":
    
    #renameImages(folder,"ed_")
    #renameImages(folder2,"miojo_")
    
    resizeImages(folder)
    resizeImages(folder2)