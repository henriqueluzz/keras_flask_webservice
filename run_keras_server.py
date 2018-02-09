from keras.preprocessing.image import img_to_array
from keras.models import model_from_json
from keras.applications import imagenet_utils
from PIL import Image
# verificar uso do scipy, ver qual é mais rapido
import io
import numpy as np
import flask
from flask import Flask, request, render_template

#curl -X POST -F image=@out.png 'http://localhost:5000/predict'

app = Flask(__name__)
model = None

'''
 if using custom model, modify the load_function to load architecture + weights
'''

def load_model():

    global model
    # load json and create model
    json_file = open('gray_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    model.load_weights("model_weights_gray_model.h5")
    print("*Modelo Carregado*")

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'Adam',
                  metrics = ['accuracy'])
    
def prepare_image(img,target):
    
    img = img.resize(target)
    img = img_to_array(img)
    img = img/255
    img = np.expand_dims(img, axis = 0)
    
    return img


@app.route("/")
def homepage():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            
            img = flask.request.files["image"].read()
            img = Image.open(io.BytesIO(img)).convert(mode = 'L')
            
            image = prepare_image(img, target = (150, 150))
            preds = model.predict(image).tolist()
            
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            r = {"label": 'ed', "probability": float(preds[0][0])}
            r2 = {"label": 'miojo', "probability": float(preds[0][1])}

            data["predictions"].append(r)
            data["predictions"].append(r2)

            # indicate that the request was a success
            data["success"] = True

            out = str(float(preds[0][1]))


    # return the data dictionary as a JSON response
    #return flask.jsonify(data['predictions'])
    return out

if __name__ == "__main__":
    print("* Loading Keras model and Flask starting server...")
    load_model()
    app.run()
                        