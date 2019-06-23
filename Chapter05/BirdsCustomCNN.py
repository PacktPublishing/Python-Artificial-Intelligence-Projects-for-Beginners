import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import itertools
# In[1]:import lib

# all images will be converted to this size
ROWS = 256
COLS = 256
CHANNELS = 3
# In[1]:import lib
train_image_generator = ImageDataGenerator(horizontal_flip=True, rescale=1./255, rotation_range=45)
test_image_generator = ImageDataGenerator(horizontal_flip=False, rescale=1./255, rotation_range=0)
# In[1]:import lib
train_generator = train_image_generator.flow_from_directory('train', target_size=(ROWS, COLS), class_mode='categorical')
test_generator = test_image_generator.flow_from_directory('test', target_size=(ROWS, COLS), class_mode='categorical')
# In[1]:import lib
train_generator.reset()
test_generator.reset()
# In[1]:import lib
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(ROWS, COLS, CHANNELS)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(400))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(200))
model.add(Activation('softmax'))
# In[1]:import lib
model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
# In[1]:import lib
model.summary()
# In[1]:import lib
tensorboard = TensorBoard(log_dir='./logs/custom')
# In[1]:import lib
model.fit_generator(train_generator, steps_per_epoch=512, epochs=10, callbacks=[tensorboard], verbose=2)

# In[1]:save model
model.save("birds-customCNN.model")

# In[1]:import lib
print(model.evaluate_generator(test_generator, steps=1000))