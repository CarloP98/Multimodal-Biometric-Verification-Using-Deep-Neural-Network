############################################### Siamese Face Recognion #############################################
from tensorflow.keras import models , optimizers , losses ,activations , callbacks
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from PIL import Image
import tensorflow as tf
import time
import os
import numpy as np

class Recognizer (object) :

	def __init__( self ):

		#tf.logging.set_verbosity( tf.logging.ERROR )

		self.__DIMEN1 = 192
		self.__DIMEN2 = 128

		input_shape = ( (self.__DIMEN1*self.__DIMEN2) * 3 , )
		convolution_shape = ( self.__DIMEN1 , self.__DIMEN2 , 3 )

		seq_conv_model = [
			Reshape( input_shape=input_shape , target_shape=convolution_shape),
			Conv2D( 32, kernel_size=(5,5) , strides=1 , activation=activations.relu ),
			MaxPooling2D(pool_size=(2,2), strides=2),
			Dropout(0.2),
			Conv2D( 32, kernel_size=(5,5), strides=1, activation=activations.relu),
			MaxPooling2D(pool_size=(2,2), strides=2 ),
			Dropout(0.2),
			Conv2D( 64, kernel_size=(5,5) , strides=1 , activation=activations.relu ),
			MaxPooling2D(pool_size=(2,2), strides=2),
			Dropout(0.2),
			Conv2D( 64, kernel_size=(5,5) , strides=1 , activation=activations.relu ),
			MaxPooling2D(pool_size=(2,2) , strides=2),
			Dropout(0.2),
			Flatten(),
			Dense( 128 , activation=activations.sigmoid )
		]
		seq_model = tf.keras.Sequential( seq_conv_model )

		input_x1 = Input( shape=input_shape )
		input_x2 = Input( shape=input_shape )

		output_x1 = seq_model( input_x1 )
		output_x2 = seq_model( input_x2 )

		distance_euclid = Lambda( lambda tensors : K.abs( tensors[0] - tensors[1] ))( [output_x1 , output_x2] )
		outputs = Dense( 1 , activation=activations.sigmoid) ( distance_euclid )
		self.__model = models.Model( [ input_x1 , input_x2 ] , outputs )

		self.__model.compile( loss=losses.binary_crossentropy , optimizer=optimizers.Adam(lr=0.0001) , metrics=['acc'] )


	def fit(self, X, Y ,  hyperparameters  ):
		initial_time = time.time()
		self.__model.fit( X  , Y ,
						 batch_size=hyperparameters[ 'batch_size' ] ,
						 epochs=hyperparameters[ 'epochs' ] ,
						 callbacks=hyperparameters[ 'callbacks'],
						 validation_data=hyperparameters[ 'val_data' ]
						 )
		final_time = time.time()
		eta = ( final_time - initial_time )
		time_unit = 'seconds'
		if eta >= 60 :
			eta = eta / 60
			time_unit = 'minutes'
		self.__model.summary( )
		print( 'Elapsed time acquired for {} epoch(s) -> {} {}'.format( hyperparameters[ 'epochs' ] , eta , time_unit ) )

	def evaluate(self , test_X , test_Y  ) :
		return self.__model.evaluate(test_X, test_Y)

	def predict(self, X  ):
		predictions = self.__model.predict( X  )
		return predictions

	def summary(self):
		self.__model.summary()

	def save_model(self , file_path ):
		self.__model.save(file_path )

	def load_model(self , file_path ):
		self.__model = models.load_model(file_path)

""" Main Model """
from tensorflow.keras.callbacks import TensorBoard
from PIL import Image
import numpy as np
import time

data_dimension1 = 192
data_dimension2 = 128
recognizer = Recognizer()
recognizer.load_model('model_2021_03_12.h5')

#####################################  Voice Recognition ###########################################
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
    file = os.path.join(dir, random.choice(os.listdir(dir)))
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
    if(max[0] < 0.55):
        return 'Unknown' , 0
    return peopleID[index] , int(max[0]*100)  ##"Voice: {0}, {1}".format(peopleID[index], int(max[0]*100))

#CREATE MODEL
s = createSpectogram(random_file('DATA/TRAIN/'))
INPUT_SHAPE = [2,s.shape[0], s.shape[1]]
model = get_siamese_model((s.shape[0], s.shape[1], 1))
optimizer = Adam(lr=LEARNING_RATE)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
model.load_weights("weights.4.h5")
#model.summary()

################################################################################################################
# Face Detection using Haar feature-based cascade classifiers:  Paul Viola and Michael Jones in their paper, "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001
from tensorflow.keras.models import Sequential
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sklearn
import cv2
import os
from threading import Thread

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
################################################# Combined Models #####################################################

image_database = ['Mohammad13.jpg','Neda05.jpg','Lenna11.jpg']#'os.listdir('Face_Database/')

#SHARED VARIABLES BETWEEN BOTH THREADS
who_is_voice = ""
who_is_voice_prob = 0.0

class Recognition():
    def Audio(self):
        global who_is_voice
        global who_is_voice_prob
        while 1:
            #start = time.time()
            recording = sd.rec(int(SECONDS * FREQUENCY), samplerate=FREQUENCY, channels=1, dtype='int16')
            sd.wait()
            audio_tensor = tf.squeeze(recording, axis=-1)
            tensor = tf.cast(audio_tensor, tf.float32) / 32768.0
            tensor = tensor[::DOWNSAMPLE]
            spectrogram = tfio.experimental.audio.spectrogram(tensor, nfft=512, window=512, stride=256).numpy()
            who_is_voice, who_is_voice_prob = testAudios(model, "TESTING", spectrogram)
            #end = time.time()
            #print(end-start)
    def Video(self):
        global who_is_voice
        global who_is_voice_prob
        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0)
        name = ''
        if vc.isOpened():  # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False
        while rval:
            cv2.imshow("preview", frame)
            rval, frame = vc.read()
            key = cv2.waitKey(100)
            img1 = frame
            img1 = cv2.resize(img1, (225, 225), interpolation=cv2.INTER_AREA)
            # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            fac1 = 5
            image1_bb = face_cascade.detectMultiScale(img1, 1.3, fac1)
            while (image1_bb == () and fac1 > 1):
                fac1 = fac1 - 1
                image1_bb = face_cascade.detectMultiScale(img1, 1.3, fac1)
            if image1_bb == ():
                image1_bb = np.array([[0, 0, 225, 225]])
            x = image1_bb[0, 0]
            y = image1_bb[0, 1]
            w = image1_bb[0, 2]
            h = image1_bb[0, 3]
            who_is_x = x
            who_is_y = y
            who_is_w = w
            who_is_h = h

            face1_cropped = img1[max(0,y-40):min(255,y+h+40), max(x-40,0):min(x+w+40,225)]
            img1_wild_test = cv2.resize(face1_cropped, (128, 192), interpolation=cv2.INTER_AREA)
            img1_wild_test = np.reshape(img1_wild_test,
                                        (1, img1_wild_test.shape[0], img1_wild_test.shape[1], img1_wild_test.shape[2]))
            who_is_this = 'Unknown'
            who_is_this_cropped_img = 0 * face1_cropped
            who_is_this_prob = 0.55

            for image_wild_2 in image_database:
                img2 = plt.imread('Face_Database/' + image_wild_2)
                """
                img2 = cv2.resize(img2, (225, 225), interpolation=cv2.INTER_AREA)
                fac2 = 5
                image2_bb = face_cascade.detectMultiScale(img2, 1.3, fac2)
                while (2 == () and fac2 > 1):
                    fac2 = fac2 - 1
                    image2_bb = face_cascade.detectMultiScale(img2, 1.3, fac2)
                if image2_bb == ():
                    image2_bb = np.array([[0, 0, 225, 225]])
                x = image2_bb[0, 0]
                y = image2_bb[0, 1]
                w = image2_bb[0, 2]
                h = image2_bb[0, 3]
                """
                face2_cropped = img2#[max(0,y-40):min(255,y+h+40), max(x-40,0):min(x+w+40,225)]
                img2_wild_test = cv2.resize(face2_cropped, (128, 192), interpolation=cv2.INTER_AREA)
                img2_wild_test = np.reshape(img2_wild_test, (
                1, img2_wild_test.shape[0], img2_wild_test.shape[1], img2_wild_test.shape[2]))
                X1_img_wild_test = img1_wild_test.reshape(
                    (img1_wild_test.shape[0], data_dimension1 * data_dimension2 * 3)).astype(np.float32)
                X2_img_wild_test = img2_wild_test.reshape(
                    (img2_wild_test.shape[0], data_dimension1 * data_dimension2 * 3)).astype(np.float32)
                Ypred_img_wild_test = recognizer.predict([X1_img_wild_test, X2_img_wild_test])
                if Ypred_img_wild_test > who_is_this_prob:
                    who_is_this_prob = Ypred_img_wild_test
                    who_is_this = image_wild_2[0:len(image_wild_2)-6]

            cv2.putText(img1, 'Image:' + who_is_this + ', ' + str(np.squeeze(int(who_is_this_prob * 100))) + "%",
                        (who_is_x - 40, who_is_y - 50), cv2.FONT_HERSHEY_SIMPLEX, min(who_is_w, who_is_h) / 200,
                        (0, 255, 0), 1)
            cv2.putText(img1, 'Voice:' + who_is_voice + ', ' + str(np.squeeze(who_is_voice_prob)) + "%",
                        (who_is_x - 40, who_is_y - 30), cv2.FONT_HERSHEY_SIMPLEX, min(who_is_w, who_is_h) / 200,
                        (0, 255, 0), 1)
            print(who_is_this)
            print(who_is_voice)
            if who_is_this == who_is_voice:
                final_detected_person = who_is_this
            else:
                final_detected_person = 'Unknown'
            cv2.putText(img1, final_detected_person, (who_is_x , who_is_y + who_is_h + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, min(who_is_w, who_is_h) / 200, (0, 255, 0), 1)
            frame = img1
            frame = cv2.rectangle(frame, (who_is_x, max(0, who_is_y - 20)), (who_is_x + who_is_w, min(who_is_y + who_is_h + 20, 255)), (0, 255, 0), 1)
            frame = cv2.resize(frame, (500, 500), interpolation=cv2.INTER_AREA)
            if key == 27:
                break
        cv2.destroyWindow("preview")

if __name__ == "__main__":
    Yep = Recognition()
    thread = Thread(target = Yep.Audio)
    thread2 = Thread(target=Yep.Video)
    thread.start()
    thread2.start()
    thread.join()



