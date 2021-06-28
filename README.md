# Rock-paper-scissors
### Problem definition and data
Rock Paper Scissors is a two-players game in which each player simultaneously forms one of three shapes with an outstretched hand. The shapes are “rock” (a closed fist), “paper” (a flat hand), and “scissors” (the index and middle fingers extended to form a “V”).  
The combination of the shapes formed by the two players determines the winner and the loser. A draw is also a possibile outcome.
We want to build a system that uses digital cameras to acquire images of the hands of the players. Images will be then automatically classified in the three forms “rock”, “paper”, and “scissors”.  
A data set has been collected by taking 2188 samples of the three forms. It is divided into a training (1888), a validation (150) and a test (150) set. The data set is organized in three directories (train, validation and test) each one including a subdirectory for each of the three classes. Each image has a resolution of 224 × 244 pixels in the RGB color space.    
<p align="center"> <img src="https://i.ibb.co/JCpFpyX/pasted-image-0.png" alt="pasted-image-0" style="display: block; margin-left: auto; margin-right: auto; padding:25%"> </p> <br>  
  
### Goal
The goal is to build a classifier that is able to recognize the three forms.  
I'll go to :  
1. analyze and comment the data;
2. design and implement a suitable data pre-processing procedure;
3. implement, train and evaluate one or more classification models;
4. use suitable data processing and visualization techniques to analyze the behavior of the trained model(s);
