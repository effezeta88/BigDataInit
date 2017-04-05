# BigDataInit
Easy example of Big Data in Python with Theano, Keras
Load file, create network model, train it, evaluate that model and predictions

from keras.models import Sequential
from keras.layers import Dense
import numpy as nmp
nmp.random.seed(7)

#load data
dataset = nmp.loadtxt("data.csv", delimiter=",")

#split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

#create networks
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#how to evaluate weight
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#train the model 150 epochs (times) all the dataset
model.fit(X,Y,epochs=150,batch_size=10)

#evaluate the model
#scores = model.evaluate(X, Y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#calculate predictions
predictions = model.predict(X)

#predictions of X and compare with Y 
p=0
for x in predictions:
	print(round(x[0]),Y[p])
	p=p+1
