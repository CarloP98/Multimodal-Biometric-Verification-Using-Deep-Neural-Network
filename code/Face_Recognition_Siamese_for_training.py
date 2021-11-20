from tensorflow.keras import models, layers, activations, losses, optimizers
import tensorflow.keras.backend as K
import tensorflow as tf
from skimage import data, transform , color
import matplotlib.pyplot as plt
import cv2
from numpy import genfromtxt
import numpy as np
import numpy as np
import os
from tensorflow import keras
from PIL import Image


training_data_path ="C:\\Users\\m_pou\\OneDrive\\Documents\\Artificial Intelligence\\Deep Learning\\Project\\Database\\colorferet\\Dataset\\"
training_data_list = os.listdir(training_data_path)

label_gt = []
group_num = [0]

#testing_data_path =
j= 0

for i in range(len(training_data_list)-1):
    if  (training_data_list[i][0:5] != training_data_list[i+1][0:5]):
        group_num.append(i+1)

g_len = len(group_num)
for j in range(g_len):
    list_others = []
    if j == 0:
        list_others= (group_num[1:g_len+1])
    if j == (g_len-1):
        list_others = (group_num[0:g_len])
    if j > 0 and j<(g_len-1):
        list_others = group_num[0:j] + group_num[j+1:g_len+1]
    if j == (g_len-1) :
        for i in range(group_num[j], len(training_data_list)-1):
            label_gt.append([training_data_list[i], training_data_list[i + 1], 1])
            label_gt.append([training_data_list[i], training_data_list[np.random.choice(list_others)], 0])
    else:
        for i in range(group_num[j],(group_num[j+1]-1)):
            label_gt.append([training_data_list[i] , training_data_list[i+1] , 1])
            label_gt.append([training_data_list[i] , training_data_list[np.random.choice(list_others)] , 0])

X_images_1 = np.zeros([len(label_gt),192,128,3],dtype = 'uint8')
X_images_2 = np.zeros([len(label_gt),192,128,3],dtype = 'uint8')
Y_label = np.zeros([len(label_gt)])

for i in range(len(label_gt)):
	X_images_1[i,:,:,:] = cv2.resize(plt.imread(training_data_path + label_gt[i][0]), (128,192), interpolation = cv2.INTER_AREA)
	X_images_2[i,:,:,:] = cv2.resize(plt.imread(training_data_path + label_gt[i][1]), (128,192), interpolation = cv2.INTER_AREA)
	Y_label[i] =label_gt[i][2]

""" Siamese Model Functions """
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
			Dropout(0.3),
			Conv2D( 32, kernel_size=(5,5), strides=1, activation=activations.relu),
			MaxPooling2D(pool_size=(2,2), strides=2 ),
			Dropout(0.3),
			Conv2D( 64, kernel_size=(5,5) , strides=1 , activation=activations.relu ),
			MaxPooling2D(pool_size=(2,2), strides=2),
			Dropout(0.3),
			Conv2D( 64, kernel_size=(5,5) , strides=1 , activation=activations.relu ),
			MaxPooling2D(pool_size=(2,2) , strides=2),
			Dropout(0.3),
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
"""Confusion Matrix"""
import numpy as np


def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:1f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

""" Main Model """
from tensorflow.keras.callbacks import TensorBoard
from PIL import Image
import numpy as np
import time

data_dimension1 = 192
data_dimension2 = 128

X1 = X_images_1[1731:20731,:,:,:]
X2 = X_images_2[1731:20731,:,:,:]
Y = Y_label[1731:20731]
X1 = X1.reshape( ( X1.shape[0]  , data_dimension1* data_dimension2 * 3  ) ).astype( np.float32 )
X2 = X2.reshape( ( X2.shape[0]  , data_dimension1* data_dimension2 * 3  ) ).astype( np.float32 )

X1_test = X_images_1[0:1731,:,:,:]
X2_test = X_images_2[0:1731,:,:,:]
Y_test = Y_label[0:1731]
X1_test_reshaped = X1_test.reshape( ( X1_test.shape[0]  , data_dimension1* data_dimension2 * 3  ) ).astype( np.float32 )
X2_test_reshaped = X2_test.reshape( ( X2_test.shape[0]  , data_dimension1* data_dimension2 * 3  ) ).astype( np.float32 )

recognizer = Recognizer()

parameters = {
    'batch_size' : 200,
    'epochs' : 50 ,
    'callbacks':  [ TensorBoard( log_dir='logs/{}'.format( time.time() ) ) ] ,
    'val_data' : ([X1_test_reshaped,X2_test_reshaped],Y_test)
}

history = recognizer.fit( [ X1 , X2 ], Y, hyperparameters= parameters)
recognizer.save_model('model_New.h5')
#load_path = "C:\\Users\\m_pou\\OneDrive\\Documents\\Artificial Intelligence\\Deep Learning\\Project\\Python_Codes\\"
#recognizer.load_model('model_2021_03_13.h5')

######### Confusion Matrix ########################
from sklearn.metrics import confusion_matrix
Y_pred = recognizer.predict([X1_test_reshaped,X2_test_reshaped])>0.5
CM = confusion_matrix(Y_test,Y_pred)
plot_confusion_matrix(CM,target_names = ['TRUE', 'FALSE'], title='Confusion matrix for Face Recognition',cmap=None,normalize=False)