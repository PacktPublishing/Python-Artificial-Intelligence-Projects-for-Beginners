from keras.models import load_model
from keras.preprocessing import image
from os import listdir
import numpy as np

ROWS = 256
COLS = 256

CLASS_NAMES = sorted(listdir('images'))

model = load_model('birds-inceptionv3.model')

def predict(fname):
    img = image.load_img(fname, target_size=(ROWS, COLS))
    img_tensor = image.img_to_array(img) # (height, width, channels)
    # (1, height, width, channels), add a dimension because the model expects this shape:
    # (batch_size, height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0) 
    img_tensor /= 255. # model expects values in the range [0, 1]
    prediction = model.predict(img_tensor)[0]
    best_score_index = np.argmax(prediction)
    bird = CLASS_NAMES[best_score_index] # retrieve original class name
    print("Prediction: %s (%.2f%%)" % (bird, 100*prediction[best_score_index]))

predict('test-birds/annas_hummingbird_sim_1.jpg')
predict('test-birds/house_wren.jpg')
predict('test-birds/canada_goose_1.jpg')

# interactive user input
while True:
    fname = input("Enter filename: ")
    if(len(fname) > 0):
        try:
            predict(fname)
        except Exception as e:
            print("Error loading image: %s" % e)
    else:
        break

