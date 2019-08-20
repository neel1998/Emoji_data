import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import os.path as osp
import os

img_dimen = 100
train_data_count = 510
test_data_count = 90
train_data = np.genfromtxt('./100_train_600.csv', delimiter=',', skip_header=0)
train_label = train_data[:,0]
train_images = train_data[:,1:]
train_data = train_images.reshape(train_data_count,img_dimen,img_dimen,1)

train_label = to_categorical(train_label)

test_data = np.genfromtxt('./100_test_600.csv', delimiter=',', skip_header=0)
test_label = test_data[:,0]
test_images = test_data[:,1:]
test_data = test_images.reshape(test_data_count,img_dimen,img_dimen,1)

es = EarlyStopping(monitor = 'val_acc', patience = 10, mode = 'max')
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                 activation='sigmoid',
                 input_shape=(img_dimen,img_dimen,1)))
model.add(MaxPooling2D(pool_size=(5, 5)))
model.add(Conv2D(64, (5, 5), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adagrad',
              metrics=['accuracy'])

model.fit(train_data, train_label, validation_split = 0.15, epochs = 50)

print(test_label)
result = model.predict(test_data)
correct = incorrect = 0
predict = []
for i in range(len(result)):
	r = np.where(result[i] == np.max(result[i]))[0][0]
	predict += [r]
	if (test_label[i] == r):
		correct += 1
	else :
		incorrect += 1
print(predict)
print(correct*100.0/(correct + incorrect))

nb_classes = 6 # The number of output nodes in the model
prefix_output_node_names_of_final_network = 'output_node'

K.set_learning_phase(0)

pred = [None]*nb_classes
pred_node_names = [None]*nb_classes
for i in range(nb_classes):
    pred_node_names[i] = prefix_output_node_names_of_final_network+str(i)
    pred[i] = tf.identity(model.output[i], name=pred_node_names[i])
print('output nodes names are: ', pred_node_names)

sess = K.get_session()
output_fld = 'tensorflow_model/'
if not os.path.isdir(output_fld):
    os.mkdir(output_fld)
output_graph_name = './saved_model_2' + '.pb'
output_graph_suffix = '_inference'

constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
graph_io.write_graph(constant_graph, output_fld, output_graph_name, as_text=False)
print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))