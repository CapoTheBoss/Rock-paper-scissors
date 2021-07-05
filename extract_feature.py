import numpy as np
import matplotlib.pyplot as plt
import os
import image_features

classes = os.listdir("2021-07-09-rock-paper-scissors/test")
classes.sort()

#some normalization method
def l2_norm(X):
	q = np.sqrt((X ** 2).sum(1, keepdims=True))
	q = np.maximum(q, 1e-15) # 1e-15 avoids division by zero 
	X = X/q

	return X


def l1_norm(X):
	q = np.abs(X).sum(1, keepdims=True)
	q = np.maximum(q, 1e-15) # 1e-15 avoids division by zero 
	X = X/q

	return X

def mu_sigma(X):
	mu = X.mean(0)
	sigma = X.std(0)
	return mu,sigma

def mean_std_norm(X, mu ,sigma):
	#feature norm
	X =(X-mu)/sigma
	return X

'''
Low lever feature function :
edge_direction_histogram
cooccurrence_matrix
rgb_cooccurrence_matrix

'''
def process_directory(path):
	all_features =[]
	all_labels = []
	class_label = 0
	for klass in classes:
		class_path = path +"/"+klass
		image_file = os.listdir(class_path)
		for imagename in image_file:
			image = plt.imread(class_path+"/"+imagename)
			image = image/255
			
			features1 = image_features.edge_direction_histogram(image)
			features2 = image_features.color_histogram(image)

			features1 = features1.reshape(-1)
			features2 = features2.reshape(-1)
			
			#conc_features = features1
			conc_features = np.concatenate((features1,features2))

			all_features.append(conc_features)
			all_labels.append(class_label)
		class_label += 1
	X = np.stack(all_features,0)
	Y = np.array(all_labels)
	return X,Y

X,Y = process_directory("2021-07-09-rock-paper-scissors/train")
print("save train",X.shape,Y.shape)
mu,sigma = mu_sigma(X)
X = mean_std_norm(X,mu,sigma)
data = np.concatenate([X,Y[:,None]],1)
np.savetxt("data/train_edge_plus_colorHist_mu_sigma.txt.gz", data)
print("saved")

X,Y = process_directory("2021-07-09-rock-paper-scissors/test")
print("test",X.shape,Y.shape)
X = mean_std_norm(X,mu,sigma)
data = np.concatenate([X,Y[:,None]],1)
np.savetxt("data/test_edge_plus_colorHist_mu_sigma.txt.gz", data)
print("saved")

X,Y = process_directory("2021-07-09-rock-paper-scissors/validation")
print("validation",X.shape,Y.shape)
X = mean_std_norm(X,mu,sigma)
data = np.concatenate([X,Y[:,None]],1)
np.savetxt("data/validation_edge_plus_colorHist_mu_sigma.txt.gz", data)
print("saved")