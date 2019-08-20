import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

img_dimen = 100
train_data_count = 255
train_data = np.genfromtxt('./100_train_600.csv', delimiter=',', skip_header=0)
train_label = train_data[:,0]
train_images = train_data[:,1:]
# train_data = train_images.reshape(train_data_count,img_dimen,img_dimen,1)

train_label = to_categorical(train_label)
# print(train_label[241])
# plt.imshow(train_data[241])
# plt.show()
# print(np.shape(train_label))

model = Sequential()
model.add(Dense(2500, input_dim = 10000, init= 'uniform', activation = 'relu'))
model.add(Dense(600, kernel_initializer='uniform', activation = 'relu'))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(train_images, train_label, validation_split = 0.15, epochs=3)
model.summary()