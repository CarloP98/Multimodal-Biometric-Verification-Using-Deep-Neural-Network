import os
import random
import numpy as np
import soundfile as sf
import tensorflow as tf
import sounddevice as sd
import tensorflow_io as tfio
from functools import partial
from tensorflow.keras import activations
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.layers import Flatten,Lambda,Dense,Dropout

#HYPERPARAMETERS
SECONDS = 3
EPOCHS = 70
DOWNSAMPLE = 3
VAL_STEPS = 100
BATCH_SIZE = 32
FREQUENCY = 16000
THRESHOLD = 800
LEARNING_RATE = 0.001
TRAIN_STEPS_PER_EPOCHS = 300
maxSamples = SECONDS * FREQUENCY

def createSpectogram(file):
    audio = tfio.audio.AudioIOTensor(file)
    audio_tensor = tf.squeeze(audio[100:], axis=-1)
    tensor = tf.cast(audio_tensor, tf.float32) / 32768.0
    position = tfio.experimental.audio.trim(tensor, axis=0, epsilon=0.1)
    start, stop = position[0], position[1]
    tensor = tensor[start:stop]

    if len(tensor) < maxSamples:
        pad = maxSamples - len(tensor)
        bef = np.random.randint(pad)
        aft = pad-bef
        if bef > 0: tensor = tf.concat([tf.zeros(bef, dtype=tf.float32),tensor], 0)
        if aft > 0: tensor = tf.concat([tensor, tf.zeros(aft, dtype=tf.float32)], 0)
    else:
        idx = random.randint(0,len(tensor)-maxSamples)
        tensor = tensor[idx:idx+maxSamples]
    tensor = tensor[::DOWNSAMPLE]
    spectrogram = tfio.experimental.audio.spectrogram(tensor, nfft=512, window=512, stride=256)
    return spectrogram.numpy()

def get_siamese_model(input_shape):
    left_input,right_input = Input(input_shape),Input(input_shape)
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Dropout(.3))
    model.add(Conv2D(64, (4, 4), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(.3))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(.3))
    model.add(Conv2D(128, (2, 2), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(2048, activation='sigmoid'))
    left = model(left_input)
    right = model(right_input)
    distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([left, right])
    output = Dense(1, activation=activations.sigmoid)(distance)
    siamese_net = Model(inputs=[left_input, right_input], outputs=output)
    return siamese_net

def random_file(dir):
    file = os.path.join(dir, random.choice(os.listdir(dir)));
    if os.path.isdir(file):return random_file(file)
    elif not file.endswith('.flac'):return random_file(dir)
    else:return file

def createData(InDir, Size, name):
    writer = tf.compat.v1.python_io.TFRecordWriter(name + ".tfrec")
    LENGTH = len(next(os.walk(InDir))[1])
    for i in range(1,Size+1):
        print(i)
        fold = random.randint(1, LENGTH)
        file = random_file(InDir + "/C (" + str(fold) + ")")
        spectogram = createSpectogram(file)
        out = [0]
        if(i%2 == 0):
            file = random_file(InDir + "/C (" + str(fold) + ")")
            spectogram2 = createSpectogram(file)
            out=[1]
        else:
            asd = random.choice([i for i in range(1, LENGTH) if i not in [fold]])
            file = random_file(InDir + "/B (" + str(asd) + ")")
            spectogram2 = createSpectogram(file)
        feature_dict = {
            'IN1': tf.train.Feature(float_list=tf.train.FloatList(value=spectogram.flatten())),
            'IN2': tf.train.Feature(float_list=tf.train.FloatList(value=spectogram2.flatten())),
            'OUT': tf.train.Feature(int64_list=tf.train.Int64List(value=out))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        writer.write(example.SerializeToString())

def read_tfrecord(example,labeled):
    tfrecord_format = (
        {
            "IN1": tf.io.FixedLenFeature([125,257], tf.float32),
            "IN2": tf.io.FixedLenFeature([125,257], tf.float32),
            "OUT": tf.io.FixedLenFeature([1], tf.int64)
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    print(example)
    in1 = tf.cast(example["IN1"], tf.float32)
    in2 = tf.cast(example["IN2"], tf.float32)
    out = tf.cast(example["OUT"], tf.int64)
    return (in1,in2), out

def getData(files):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(partial(read_tfrecord, labeled=True), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

def generate_data(directory, batch_size):
    LENGTH = len(next(os.walk(directory))[1])
    while True:
        batch1,batch2, batchOut = [],[],[]
        for b in range(batch_size):
            fold = random.randint(1, LENGTH)
            file = random_file(directory + "/C (" + str(fold) + ")")
            spectogram = createSpectogram(file)
            out = [0]
            if (b % 2 == 0):
                file = random_file(directory + "/C (" + str(fold) + ")")
                spectogram2 = createSpectogram(file)
                out = [1]
            else:
                asd = random.choice([i for i in range(1, LENGTH) if i not in [fold]])
                file = random_file(directory + "/C (" + str(asd) + ")")
                spectogram2 = createSpectogram(file)
            batch1.append(spectogram)
            batch2.append(spectogram2)
            batchOut.append(out)
        yield [np.asarray(batch1),np.asarray(batch2)], np.asarray(batchOut)


def recordAudio(name, Path):
    recording = sd.rec(int((SECONDS) * FREQUENCY), samplerate=FREQUENCY, channels=1, dtype='int16')
    sd.wait()
    sf.write(Path + "/" + name + '.flac', recording, FREQUENCY)

def testAudios(model, path, recording):
    peopleID, batch1, batch2 = [], [], []
    for filename in os.listdir(path):
        peopleID.append(filename.rsplit('.', 1)[0])
        batch1.append(recording)
        batch2.append(createSpectogram(path + "/" + filename))
    predictions = model.predict([np.asarray(batch1), np.asarray(batch2)])
    index = np.argmax(predictions)
    max = predictions[index]
    if(max[0] < 0.5):
        return "None"
    return "Person: {0}, Probability: {1}".format(peopleID[index], max[0])

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

#CREATE MODEL
voice_path = "C:\\Users\\m_pou\\OneDrive\\Documents\\Artificial Intelligence\\Deep Learning\\Project\\Python_Codes\\"
s = createSpectogram(random_file(voice_path+'DATA\TRAIN'))
INPUT_SHAPE = [2,s.shape[0], s.shape[1]]
model = get_siamese_model((s.shape[0], s.shape[1], 1))
optimizer = Adam(lr=LEARNING_RATE)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
model.load_weights(voice_path + 'weights.4.h5')
#model.summary()

#CREATE AUDIO
#recordAudio("Mohammad","TESTING")
import cv2

while 1:
    #RECORD
    print("Recording...")
    recording = sd.rec(int(SECONDS * FREQUENCY), samplerate=FREQUENCY, channels=1, dtype='int16')
    sd.wait(100)
    #SKIP IF TOO QUIET
    if(max(recording)< THRESHOLD):
        print("Too quiet")
        continue
    #CREATE SPECTOGRAM
    print("Evaluating")
    audio_tensor = tf.squeeze(recording, axis=-1)
    tensor = tf.cast(audio_tensor, tf.float32) / 32768.0
    tensor = tensor[::DOWNSAMPLE]
    spectrogram = tfio.experimental.audio.spectrogram(tensor, nfft=512, window=512, stride=256).numpy()
    #TEST
    print(testAudios(model,"TESTING",spectrogram))
