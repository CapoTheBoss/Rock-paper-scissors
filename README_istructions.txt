To successfully execute the set of final results reported in the report, follow these steps:
- adding pvmlnet.npz
- add the folder with photos
- run extract_neural_fetaures.py
- run extract_fetaures.py
- run train_classifier.py 4 times to create:
	- 2 with the neural features -> cnn and cnn_128
	- 2 with the low level features -> mlp and mlp_128
- run evaluate_model.py

For intermediate results, manually change the low level features and normalizations