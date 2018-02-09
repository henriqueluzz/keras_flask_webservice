from keras.models import model_from_json
from keras.preprocessing import image
import json
import cv2
import numpy as np
import os

'''
{'ed':0, 'miojo':1}

'''
def load_images(img):

    img = image.load_img(img, target_size = (150,150), grayscale = True)
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)

    return x

def batch_predict():

    for fileName in os.listdir("./img/validation/miojo/"):
    
        aux = dict()
        img = load_images("./img/validation/miojo/"+fileName)
        y_prob = model.predict(img).tolist()

        aux['ed'] = y_prob[0][0]
        aux['miojo'] = y_prob[0][1]
        
        #print(json.dumps(aux))
        print(aux)

if __name__ == "__main__":

    # load json and create model
    json_file = open('gray_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    # load weights into new model
    model.load_weights("model_weights_gray_model.h5")
    print("Loaded model from disk")

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'Adam',
                  metrics = ['accuracy'])

    batch_predict()

    #pred = model.predict_classes(load_images("out.png"))
    #print(pred)
        