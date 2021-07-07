import numpy as np
import pvml
import os
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# load image validation names
img_val = np.load("data/img_val.npz")
img_val = img_val["arr_0"]


def confusion_matrix(X,Y,predictions):

	classes = os.listdir("2021-07-09-rock-paper-scissors/validation")
	classes.sort()

	nclasses = Y.max() + 1

	cm = np.zeros((nclasses,nclasses))
	for i in range(X.shape[0]):
		cm[Y[i],predictions[i]] += 1

	cm = cm / cm.sum(1, keepdims=True)
	plt.imshow(cm)
	for i in range(nclasses):
		for j in range(nclasses):
			plt.text(j,i,int(100*cm[i,j]),color="red",size=12)
	plt.xticks(range(nclasses),classes[:],rotation=90)
	plt.yticks(range(nclasses),classes[:])
	plt.show()

def get_worst_error(img_val,Y_val,predictions,probs,text):
	
	classes = os.listdir("2021-07-09-rock-paper-scissors/validation")
	classes.sort()

	miss_classified_index = []
	max_prob = 0

	miss_classified_index = [Y_val != predictions]
	Y_val_real = Y_val[miss_classified_index]
	Y_val_pred = predictions[miss_classified_index]

	img_val = img_val[miss_classified_index]
	miss_probs = probs[miss_classified_index]
	max_prob = 0

	for x in range(miss_probs.shape[0]):
		for j in range(miss_probs.shape[1]):
			if miss_probs[x][j] > max_prob:
				max_prob = miss_probs[x][j]
				index = x


	print(text," real form :",classes[Y_val_real[x]]," predicted form :",classes[Y_val_pred[x]]," probability = ",max_prob*100,"% see image = ",img_val[x],"\n")
	

# low level features
data = np.loadtxt("data/train_edge_plus_colorHist_mu_sigma.txt.gz")
X_train_llf = data[:,:-1]
Y_train_llf = data[:,-1].astype(int)

data = np.loadtxt("data/validation_edge_plus_colorHist_mu_sigma.txt.gz")
X_val_llf = data[:,:-1]
Y_val_llf = data[:,-1].astype(int)

# neural features
data = np.loadtxt("data/train_neural_features.txt.gz")
X_train_nf = data[:,:-1]
Y_train_nf = data[:,-1].astype(int)

data = np.loadtxt("data/validation_neural_features.txt.gz")
X_val_nf = data[:,:-1]
Y_val_nf = data[:,-1].astype(int)


mlp_nhl_llf = pvml.MLP.load("models/mlp.npz") # nhl stand for 'no hidden layer'
mlp_nhl_nf = pvml.MLP.load("models/cnn.npz")

mlp_hl_llf = pvml.MLP.load("models/mlp_128.npz") # hl stand for 'hidden layer'
mlp_hl_nf = pvml.MLP.load("models/cnn_128.npz")


accuracy_value = []  # create this variable in order to plot the results
accuracy_model = []	 # create this variable in order to plot the results

# SVM with kernel radial base
kparam = 1e-2
w,b = pvml.one_vs_one_ksvm_train(X_train_llf, Y_train_llf,'rbf',kparam,lambda_= 0, lr=1e-2, steps=1000, init_alpha=None)
label,logit = pvml.one_vs_one_ksvm_inference(X_val_llf,X_train_llf,w,b,'rbf', kparam)
accuracy=(Y_val_llf==label).mean()
accuracy_value.append(accuracy*100)
accuracy_model.append("SVM llf")
print("accuracy SVM radial base low level features = ",accuracy*100,"%\n")

kparam = 3e-2
w,b = pvml.one_vs_one_ksvm_train(X_train_nf , Y_train_nf,'rbf',kparam,lambda_= 0, lr=1e-2, steps=1000, init_alpha=None)
label,logit = pvml.one_vs_one_ksvm_inference(X_val_nf,X_train_nf,w,b,'rbf', kparam)
accuracy=(Y_val_nf==label).mean()
print("accuracy SVM radial base neural features = ",accuracy*100,"%\n")
accuracy_value.append(accuracy*100)
accuracy_model.append("SVM nf")


# MLP with no hidden layer

predictions,probs = mlp_nhl_llf.inference(X_val_llf)
val_acc = (predictions==Y_val_llf).mean()
print("accuracy MLP with no hidden layer low level features = ",val_acc*100,"%")
get_worst_error(img_val,Y_val_llf,predictions,probs,"MLP no hidden llf")
accuracy_value.append(val_acc*100)
accuracy_model.append("MLP nh llf")


predictions,probs = mlp_nhl_nf.inference(X_val_nf)
val_acc = (predictions==Y_val_nf).mean()
print("accuracy MLP with no hidden layer neural features = ",val_acc*100,"%")
get_worst_error(img_val,Y_val_llf,predictions,probs,"MLP no hidden nf")
accuracy_value.append(val_acc*100)
accuracy_model.append("MLP nh nf")
confusion_matrix(X_val_nf,Y_val_nf,predictions)



# MLP with hidden layer

predictions,probs = mlp_hl_llf.inference(X_val_llf)
val_acc = (predictions==Y_val_llf).mean()
print("accuracy MLP with hidden layer low level features = ",val_acc*100,"%")
get_worst_error(img_val,Y_val_llf,predictions,probs,"MLP hidden 128 llf")
accuracy_value.append(val_acc*100)
accuracy_model.append("MLP hl llf")

predictions,probs = mlp_hl_nf.inference(X_val_nf)
val_acc = (predictions==Y_val_nf).mean()
print("accuracy MLP with hidden layer neural features = ",val_acc*100,"%")
accuracy_value.append(val_acc*100)
accuracy_model.append("MLP hl nf")
get_worst_error(img_val,Y_val_llf,predictions,probs,"MLP hidden 128 nf")

# Plot data 


fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(range(len(accuracy_model)), accuracy_value, 'bo') # Plotting data
plt.xticks(range(len(accuracy_model)), accuracy_model) # Redefining x-axis labels

for i, v in enumerate(accuracy_value):
    ax.annotate(str(int(v)), xy=(i,v), xytext=(-7,7), textcoords='offset points')
plt.ylim(0,140)
plt.show()


