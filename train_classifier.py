import numpy as np
import matplotlib.pyplot as plt
import pvml
import os

classes = os.listdir("2021-07-09-rock-paper-scissors/validation")
classes.sort()

def img_test_name(path):
	img_test = []
	for klass in classes:
		class_path = path +"/"+klass
		image_file = os.listdir(class_path)
		for imagename in image_file:
			img_test.append(imagename)
	return img_test


#test_edge_plus_colorHist_mu_sigma --> this is the test file for low level features
#train_edge_plus_colorHist_mu_sigma --> this is the train file for low level features

#test_neural_features.txt.gz --> this is the test file for neural features
#train_neural_features.txt.gz --> this is the train file for neural features

img_test = img_test_name("2021-07-09-rock-paper-scissors/test")

data = np.loadtxt("data/train_edge_plus_colorHist_mu_sigma.txt.gz")
X_train = data[:,:-1]
Y_train = data[:,-1].astype(int)

data = np.loadtxt("data/test_edge_plus_colorHist_mu_sigma.txt.gz")
X_test = data[:,:-1]
Y_test = data[:,-1].astype(int)

nclasses = Y_train.max() + 1

mlp = pvml.MLP([X_train.shape[1],nclasses])

epochs = 3000
batch_size = 50
lr = 0.0001
steps = X_train.shape[0]//batch_size

for epoch in range (epochs):	
	mlp.train(X_train,Y_train,lr=lr,batch = batch_size,steps=steps)
	if epoch % 100 == 0:
		predictions,probs = mlp.inference(X_train)
		train_acc = (predictions==Y_train).mean()
		predictions,probs = mlp.inference(X_test)
		test_acc = (predictions==Y_test).mean()
		print(epoch,train_acc*100,test_acc*100)

'''
name model 
mlp
cnn

mlp_128 for 1 hidden layer with 128 neurons
cnn_128
'''

mlp.save("models/mlp.npz") # --> uncomment if you want to save the model
