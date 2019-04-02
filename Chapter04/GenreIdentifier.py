import librosa
import librosa.feature
import librosa.display
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

# In[1]: buat fungsi mfcc untuk ngetest ajah
    
def display_mfcc(song):
    y, _ = librosa.load(song)
    mfcc = librosa.feature.mfcc(y)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.title(song)
    plt.tight_layout()
    plt.show()

# In[2]: cek fungsi
display_mfcc('266093__stereo-surgeon__kick-loop-5.wav')

# In[2]: cek fungsi
display_mfcc('98195__grrlrighter__whistling.wav')

# In[2]: cek fungsi
display_mfcc('genres/disco/disco.00035.au')
    
# In[2]: cek fungsi
display_mfcc('genres/disco/disco.00070.au')

# In[2]: cek fungsi
display_mfcc('genres/classical/classical.00067.au')

# In[2]: cek fungsi
display_mfcc('genres/hiphop/hiphop.00098.au')

# In[2]: cek fungsi
display_mfcc('genres/hiphop/hiphop.00028.au')

# In[3]: fitur ektraksi


def extract_features_song(f):
    y, _ = librosa.load(f)

    # get Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y)
    # normalize values between -1,1 (divide by max)
    mfcc /= np.amax(np.absolute(mfcc))

    return np.ndarray.flatten(mfcc)[:25000]

# In[3]: fitur ektraksi

def generate_features_and_labels():
    all_features = []
    all_labels = []

    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    for genre in genres:
        sound_files = glob.glob('genres/'+genre+'/*.au')
        print('Processing %d songs in %s genre...' % (len(sound_files), genre))
        for f in sound_files:
            features = extract_features_song(f)
            all_features.append(features)
            all_labels.append(genre)

    # convert labels to one-hot encoding cth blues : 1000000000 classic 0100000000
    label_uniq_ids, label_row_ids = np.unique(all_labels, return_inverse=True)#ke integer
    label_row_ids = label_row_ids.astype(np.int32, copy=False)
    onehot_labels = to_categorical(label_row_ids, len(label_uniq_ids))#ke one hot
    return np.stack(all_features), onehot_labels

# In[3]: passing parameter dari fitur ekstraksi menggunakan mfcc

features, labels = generate_features_and_labels()
# In[3]: fitur ektraksi
print(np.shape(features))
print(np.shape(labels))
# In[3]: fitur ektraksi
training_split = 0.8
# In[3]: fitur ektraksi
# last column has genre, turn it into unique ids
alldata = np.column_stack((features, labels))
# In[3]: fitur ektraksi
np.random.shuffle(alldata)
splitidx = int(len(alldata) * training_split)
train, test = alldata[:splitidx,:], alldata[splitidx:,:]
# In[3]: fitur ektraksi
print(np.shape(train))
print(np.shape(test))
# In[3]: fitur ektraksi
train_input = train[:,:-10]
train_labels = train[:,-10:]
# In[3]: fitur ektraksi
test_input = test[:,:-10]
test_labels = test[:,-10:]
# In[3]: fitur ektraksi
print(np.shape(train_input))
print(np.shape(train_labels))

# In[3]: membuat seq NN, layer pertama dense dari 100 neurons
model = Sequential([
    Dense(100, input_dim=np.shape(train_input)[1]),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
    ])
# In[3]: fitur ektraksi
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
# In[3]: fitur ektraksi
model.fit(train_input, train_labels, epochs=10, batch_size=32,
          validation_split=0.2)
# In[3]: fitur ektraksi
loss, acc = model.evaluate(test_input, test_labels, batch_size=32)
# In[3]: fitur ektraksi
print("Done!")
print("Loss: %.4f, accuracy: %.4f" % (loss, acc))

# In[3]: fitur ektraksi
model.predict(test_input[:1])