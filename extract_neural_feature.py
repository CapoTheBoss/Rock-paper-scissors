import numpy as np
import matplotlib.pyplot as plt
import os
import image_features
import pvml

classes = os.listdir("2021-07-09-rock-paper-scissors/test")
classes.sort()

def extract_neural_features(im , net ,level):
	im = im[None, :, :, :]
	activations = net.forward(im)
	features = activations[- level] #this is the last hidden layer 
	features = features[0,0,0,:]
	return features

def process_directory(path , net , level):
	all_features =[]
	all_labels = []
	images = []
	class_label = 0
	for klass in classes:
		class_path = path +"/"+klass
		image_file = os.listdir(class_path)
		for imagename in image_file:
			images.append(imagename)
			image = plt.imread(class_path+"/"+imagename)
			image = image/255
			features = extract_neural_features(image,net,level)
			all_features.append(features)
			all_labels.append(class_label)
		class_label += 1
	np.savez("data/img_test.npz",images)
	X = np.stack(all_features,0)
	Y = np.array(all_labels)
	return X,Y

net= pvml.CNN.load("pvmlnet.npz")
'''
X,Y = process_directory("2021-07-09-rock-paper-scissors/train", net , 3)
print("train",X.shape,Y.shape)
data = np.concatenate([X,Y[:,None]],1)
np.savetxt("data/train_neural_features.txt.gz", data)
'''

X,Y = process_directory("2021-07-09-rock-paper-scissors/test", net , 3)
print("test",X.shape,Y.shape)
data = np.concatenate([X,Y[:,None]],1)
np.savetxt("data/test_neural_features.txt.gz", data)

X,Y = process_directory("2021-07-09-rock-paper-scissors/validation", net , 3)
print("validation",X.shape,Y.shape)
data = np.concatenate([X,Y[:,None]],1)
np.savetxt("data/validation_neural_features.txt.gz", data)