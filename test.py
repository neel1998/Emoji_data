import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np

test_data = np.genfromtxt('./online_data/09_06_2019.txt', delimiter=',', skip_header=0)
# test_data = np.genfromtxt('./try.csv', delimiter=',', skip_header=0)
labels = test_data[:,0]
images = test_data[:,1:].reshape(41,100,100,1)

f = gfile.FastGFile("./tensorflow_model/saved_model_80.pb", 'rb')
graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
f.close()
sess = tf.Session()
sess.graph.as_default()
tf.import_graph_def(graph_def)
softmax_tensor = sess.graph.get_tensor_by_name('import/dense_1/Softmax:0')
predictions = sess.run(softmax_tensor, {'import/conv2d_1_input:0': images})

correct = incorrect = 0;
for i in range(len(predictions)):
	if labels[i] == np.where(predictions[i] == np.max(predictions[i]))[0][0]:
		correct += 1
	else :
		incorrect +=1
print (correct, incorrect, correct*100.0/(correct + incorrect))			
# print (np.where(predictions[0] == np.max(predictions[0]))[0][0])

