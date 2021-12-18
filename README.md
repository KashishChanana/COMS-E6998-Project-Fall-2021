# COMS-E6998-Project-Fall-2021

**Project Description:**

There has been extensive research on how to build computer vision models using image data. Detecting objects in images, classifying those objects, generating labels. Several image classification models like VGG, Inception and ResNet have been extensively studied and employed.

For this project, we decided to turn our attention to the less-heralded aspect of computer vision – videos! We are consuming video content at an unprecedented pace through social media. We feel this area of computer vision holds a lot of potential for data scientists and machine learning engineers. By being able to analyze video streams, we can obtain valuable insights to applications like surveillance, self-driving vehicles, industrial automation, and other high impact fields.

With this project, we look forward to performing Human Activity Recognition (HAR) in videos. We’ll be employing the UCF101 dataset and walking through the three approaches to do HAR, namely - 3D Convolution Architecture, Convolutional Neural Network - Recurrent Neural Network architecture ,and Convolutional Neural Network - Transformer architecture.
We experiment with differing hyperparameters of batch size, dropout, number of layers/parameters and have collected logs of 32 models. We then compare the performance of the three approaches using performance evaluation metrics of - training accuracy, training loss, validation accuracy and validation loss.

For the purpose of this project, we use the UCF 101 dataset that is linked here. 
UCF101 is an action recognition data set of realistic action videos, collected from YouTube, having 101 action categories. It consists of 13320 videos and  gives the largest diversity in terms of actions and with the presence of large variations in camera motion, object appearance and pose, object scale, viewpoint, cluttered background, illumination conditions, etc.

Some categories include Apply Eye Makeup, Apply Lipstick, Archery, Baby Crawling, Balance Beam, Band Marching, Baseball Pitch.

**Implementation Details:**

Hardware - NVIDIA T4 (16 GB), 4 CPUs, 15 GB RAM, 100 GB Persistent Storage

Cloud Platform - Google Cloud

Frameworks - Keras (with Tensorflow backend)

Libraries - OpenCV, NumPy, Pandas, Matplotlib, Scikit Learn

Prominent Layers/Modules:  3D-CNN Architecture: 3D Convolutional Layer

Shared CNN-RNN Architecture: Time Distributed Layer, LSTM

Shared CNN-Transformer Architecture: Multi-Head Attention Module

Design Choices -  We set the number of frames to be extracted per video to 10 with an interval of 5 frames. We also experimented with 20 frames with a 5 frame interval, but convergence time was too high.

Hyperparameters - Intervals/Frame, Batch Sizes, Dropout, Recurrent Layers, Units/Recurrent Layer

We collected logs from by running permutations of  the following choices of hyperparameters -
Dropout = {0.1, 0.2}; Batch Size = {8, 16}; Number of layers = {1, 2}; Number of Units = {128, 256};


**Repository Description:**



**Command to execute:**



**Results:**
![Accuracy Analysis- I](https://github.com/KashishChanana/COMS-E6998-Project-Fall-2021/blob/main/assets/Accuracy-%20I.jpg)
![Accuracy Analysis- II](https://github.com/KashishChanana/COMS-E6998-Project-Fall-2021/blob/main/assets/Accuracy-%20II.jpg)
![Loss Anlysis - I](https://github.com/KashishChanana/COMS-E6998-Project-Fall-2021/blob/main/assets/Loss%20-I.jpg)
![Loss Analysis - II](https://github.com/KashishChanana/COMS-E6998-Project-Fall-2021/blob/main/assets/Loss%20-II.jpg)
![Time To Accuracy Analysis- I](https://github.com/KashishChanana/COMS-E6998-Project-Fall-2021/blob/main/assets/TTA-%20I.jpg)
![Time To Accuracy Analysis- II](https://github.com/KashishChanana/COMS-E6998-Project-Fall-2021/blob/main/assets/TTA-%20II.jpg)
**Conclusion:**

Through our experimental evaluations, we concluded the following:
1. 3D-CNNs are expensive to train compared to 2D-CNNs due to the extra temporal dimension. Additionally, as we increase the no. of consecutive frames to consider, the no. of parameters grow exponentially.

2. CNN-RNNs show a better performance in terms of no. of parameters and training time. However, as no. of consecutive frames start to increase, the unfolding operations required also increase linearly. If we stack layers, they start increasing exponentially. Additionally, RNNs struggle to cope with longer sequences.

3. CNN-Transformers converge quicker and have a higher tolerance to overfitting even with higher no. of parameters. These models are able to handle longer sequences with ease. More transformer layers can be stacked to obtain better results.


