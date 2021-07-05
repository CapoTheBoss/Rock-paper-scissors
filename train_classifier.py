import numpy as np
import matplotlib.pyplot as plt
import pvml
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# get image test name in order to find the worst errors

classes = os.listdir("2021-07-09-rock-paper-scissors/test")
classes.sort()

def img_test_name(path):
	img_test = []
	for klass in classes:
		class_path = path +"/"+klass
		image_file = os.listdir(class_path)
		for imagename in image_file:
			img_test.append(imagename)
	return img_test


#test_edge_plus_colorHist_mu_sigma

img_test = img_test_name("2021-07-09-rock-paper-scissors/test")

data = np.loadtxt("data/train_edge_plus_colorHist_mu_sigma.txt.gz")
X_train = data[:,:-1]
Y_train = data[:,-1].astype(int)

data = np.loadtxt("data/test_edge_plus_colorHist_mu_sigma.txt.gz")
X_test = data[:,:-1]
Y_test = data[:,-1].astype(int)

nclasses = Y_train.max() + 1
classes = os.listdir("2021-07-09-rock-paper-scissors/test")
classes.sort()

lr = 3e-2
regolarizz = 0
steps = 1000

'''
w,b=pvml.logreg_train(X_train,Y_train,regolarizz,lr,steps)
label,logit = pvml.logreg_inference(X_test,w,b)
Predictions= (logit>0.5)
accuracy=(Y_test==Predictions).mean()
print(accuracy)
'''
mlp = pvml.MLP([X_train.shape[1],nclasses])

epochs = 1000
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
miss_classified_index = []
max_prob = 0

w = [Y_test != predictions]

img_test = img_test[w]
Y_miss = Y_test[w]
miss_probs = probs[w]
Y_test_miss = Y_test[w]
y_pred_miss = predictions[w]
maxprobs = []

for x in range(miss_probs.shape[0]):
	max_prob = 0
	for j in range(miss_probs.shape[1]):
		if miss_probs[x][j] > max_prob:
			max_prob = miss_probs[x][j]
	maxprobs.append(max_prob)


for i in range(len(maxprobs)):
	if maxprobs[i] > 0.95:
		print("real form :",classes[Y_test_miss[i]]," predicted form :",classes[y_pred_miss[i]]," probability = ",maxprobs[i]*100," see image = ",img_test[i])



print(np.bincount(Y_miss))
'''
cm = np.zeros((nclasses,nclasses))
for i in range(X_test.shape[0]):
		cm[Y_test[i],predictions[i]] += 1

cm = cm / cm.sum(1, keepdims=True)
plt.imshow(cm)
for i in range(nclasses):
	for j in range(nclasses):
		plt.text(j,i,int(100*cm[i,j]),color="red",size=12)
plt.xticks(range(nclasses),classes[:],rotation=90)
plt.yticks(range(nclasses),classes[:])
plt.show()

mlp.save("mlp.npz")
