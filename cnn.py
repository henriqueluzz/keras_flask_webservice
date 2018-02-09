'''
@henrique

Convolutional Neural Network for Miojo identification
'''

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import RemoteMonitor
from keras import callbacks
from keras import optimizers

remote = callbacks.TensorBoard(log_dir = './logs',
                                    histogram_freq = 0, 
                                    write_graph = True, 
                                    write_images = True)

train_dir = './img/train'
validation_dir = './img/validation'

image_size = (150,150,1)
learning_rate = 0.0005
batch_size = 32
epochs = 150

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = image_size, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.3))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.3))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.3))

classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 2, activation = 'softmax'))
#-----------------------------------------------------------

adam = optimizers.Adam(lr = learning_rate)
classifier.compile(optimizer = adam,
                   loss = 'categorical_crossentropy',
                   metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
#Image Augmentation
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = False)

test_datagen = ImageDataGenerator(rescale = 1./255)

#target_size = expected image size from CNN model
training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size = (150,150),
                                                 color_mode = "grayscale",
                                                 batch_size = batch_size,
                                                 class_mode='categorical')
#test set
test_set = test_datagen.flow_from_directory(validation_dir,
                                            target_size = (150,150),
                                            color_mode = "grayscale",
                                            batch_size = batch_size,
                                            class_mode = 'categorical')

history = classifier.fit_generator(
                        training_set,
                        steps_per_epoch = 6,
                        epochs = epochs,
                        validation_steps = 2,
                        callbacks = [remote]
                        )

classifier.save_weights('model_weights_gray_model.h5')
# serialize model to JSON

model_json = classifier.to_json()
with open("gray_model.json", "w") as json_file:
    json_file.write(model_json)

#img = cv2.imread("out.png")
#img = np.expand_dims(img,axis=0)
#classifier.predict(img)

