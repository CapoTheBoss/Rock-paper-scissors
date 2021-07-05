import numpy as np
import pvml
import matplotlib.pyplot as plt

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


mlp_nhl_llf = pvml.MLP.load("mlp.npz") # nhl stand for 'no hidden layer'
mlp_nhl_nf = pvml.MLP.load("cnn.npz")

mlp_hl_llf = pvml.MLP.load("mlp_128.npz") # hl stand for 'hidden layer'
mlp_hl_nf = pvml.MLP.load("cnn_128.npz")


# SVM with kernel radial base
kparam = 1e-2
w,b = pvml.one_vs_one_ksvm_train(X_train_llf, Y_train_llf,'rbf',kparam,lambda_= 0, lr=1e-2, steps=1000, init_alpha=None)
label,logit = pvml.one_vs_one_ksvm_inference(X_val_llf,X_train_llf,w,b,'rbf', kparam)
accuracy=(Y_val_llf==label).mean()
print("accuracy SVM radial base low level feature = ",accuracy*100,"%")

kparam = 3e-2
w,b = pvml.one_vs_one_ksvm_train(X_train_nf , Y_train_nf,'rbf',kparam,lambda_= 0, lr=1e-2, steps=1000, init_alpha=None)
label,logit = pvml.one_vs_one_ksvm_inference(X_val_nf,X_train_nf,w,b,'rbf', kparam)
accuracy=(Y_val_nf==label).mean()
print("accuracy SVM radial base neural feature = ",accuracy*100,"%")

# MLP with no hidden layer

predictions,probs = mlp_nhl_llf.inference(X_val_llf)
val_acc = (predictions==Y_val_llf).mean()
print("accuracy MLP with no hidden layer low level feature = ",val_acc*100,"%")


predictions,probs = mlp_nhl_nf.inference(X_val_nf)
val_acc = (predictions==Y_val_nf).mean()
print("accuracy MLP with no hidden layer neural feature = ",val_acc*100,"%")

# MLP with hidden layer
predictions,probs = mlp_hl_llf.inference(X_val_llf)
val_acc = (predictions==Y_val_llf).mean()
print("accuracy MLP with hidden layer low level feature = ",val_acc*100,"%")


predictions,probs = mlp_hl_nf.inference(X_val_nf)
val_acc = (predictions==Y_val_nf).mean()
print("accuracy MLP with hidden layer neural feature = ",val_acc*100,"%")
