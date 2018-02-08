# **Behavioral Cloning** 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

The goal of this project is to build a neural network that models steering angles needed for a simulated vehicle to drive itself autonomously on a racetrack. The network is trained on images and steering data, and outputs steering angles to be used by the car. The provided simulator was used to collect additional data of good driving behavior.


[//]: # (Image References)

[image1]: ./images/nvidia_cnn.png "Model Visualization"
[image2]: ./images/center.png "Center"
[image3]: ./images/sway1.png "Recovery 1"
[image4]: ./images/sway2.png "Recovery 2"
[image5]: ./images/sway3.png "Recovery 3"
[image6]: ./images/sway4.png "Recovery 4"
[image7]: ./images/flip_pre.png "Original Image"
[image8]: ./images/flip_post.png "Flipped Image"



Model Architecture
---

The model used in this project is based on the CNN model presented in [End-to-End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) from NVidia, with slight modifications.

The model consists of five convolution layers, followed by fully connected layers that regresses to a single predicted value of steering. The first three convolution layers are each 5x5 kernels with 2x2 strides, whereas the last two convolution layers are 3x3 kernels with 1x1 strides. Each convolution layer has a RELU activation to introduce non-linearities.

The outputs of the last convolution layer is flattened and are fed into a series of fully connected layers. Each layer output employs a dropout probability of 0.5.

Below is a visualization of the network, taken from the aforementioned paper.

![cnn][image1]


The final model is summarized as follows:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						| 
| Normalization       	| Normalizes input images  						|
| Cropping		     	| Input: 160x320x3; Output: 80x320x3		 	|
| Convolution 5x5     	| 24 5x5 kernels, 2x2 stride					|
| RELU 					|												|
| Convolution 5x5     	| 36 5x5 kernels, 2x2 stride					|
| RELU 					|												|
| Convolution 5x5     	| 48 5x5 kernels, 2x2 stride					|
| RELU 					|												|
| Convolution 3x3     	| 64 5x5 kernels, 1x1 stride					|
| RELU 					|												|
| Convolution 3x3     	| 64 5x5 kernels, 1x1 stride					|
| RELU 					|												|
| Flatten			    | 												|
| Fully connected		| Input: 1164, Output: 100						|
| RELU					| 												|
| Dropout				| Drop rate: 0.5								|
| Fully connected		| Input: 100, Output: 50						|
| RELU 					|												|
| Dropout				| Drop rate: 0.5								|
| Fully connected		| Input: 50, Output: 10							|
| Fully connected		| Input: 10, Output: 1							|
| Output 				| Steering value prediction						|


#### Attempts to Reduce Overfitting

[Dropout](https://keras.io/layers/core/#dropout) layers were included in the network within the fully connected layers. Not only does this implicitly induce ensemble learning in the model, it helps avoid overfitting by acting as a form of regularization. During training, dropout rate of 0.5 was used. 

[Early Stopping](https://keras.io/callbacks/#earlystopping) was another technique used to avoid overfitting. `model.fit` method can take in callback functions that get executed at the end of each epoch. Early stopping checks for various conditions to decide whether to continue training into the new epoch. For the training of this model, the validation loss was monitored with a patience of 2 epochs (i.e, stop training when 2 epochs of no improvements in validation loss). This was used in conjunction with [ModelCheckpoint](https://keras.io/callbacks/#modelcheckpoint) so that the best intermediate model so far can be restored. 


#### Model parameter tuning

The model uses the Adam optimizer, which automatically computes adaptive learning rates ([reference](http://ruder.io/optimizing-gradient-descent/index.html#adam)). Thus, the learning rate was not manually tuned. 

The use of Early stopping mitigated the need for searching the best epoch count. The value for dropout rate was set based on empirical results.



Training Data 
---

The training data used is composed of the provided sample data for track1, as well as additional training data created using the training mode of the simulator. The center, left, and right images of the car are used during training.

The data collected from the simulator are composed of several runs of recording. The following styles of training data were collected:

- __Good Driving__ : In these samples, the car maintains alignment with the center of the road, as much as possible.
- __Examples of Recovery__: These samples contain cases in which the car recovers from going off of the road by applying steering when too close to the outer edges of the road. This will help the model identify situations in which proper steering is required. 
- __Reverse traversal__ of the track: While the simulation starts out with a counter-clockwise orientation on the road, providing a clock-wise driving around the track will help generalze better and provide more examples.

Furthermore, additional examples were created by mirroring each available image and steering measurement pair. 



Solution Design Approach
---

This section lists a bit of detail regarding the iterative process taken to build the model.

The first attempt was based on [LeNet-5](http://yann.lecun.com/exdb/lenet/), but it was not able to handle some of the first few curves. This led to a collection of more training samples as previously mentioned. 

A common theme was for the validation loss to oscillate while the training loss kept decreasing. This indicated overfitting, thus Dropout and Early stopping was introduced. 

While the vehicle was struggling to complete a full lap, the addition of various techniques such as cropping the image to the region of interest, the use of left and right side cameras on top of the center image, and data augmentation led to the vehicle being able to get close to almost completeing a full lap.

Eventually, I transitioned to using the CNN from the NVidia paper. This model seemed to provide better results empirically.

Additional data from a few more simulation runs were added, since in some cases the car was getting too close to the edges. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. A footage of this is available in the repo [here](https://github.com/enju244/CarND-Behavioral-Cloning/blob/master/video.mp4).




Creation of the Training Set & Training Process
---

As mentioned in the "Training Data" section, samples were generated from simulations of various styles, including good driving behavior, recoveries to center, and driving the course in reverse.

A few laps were simulated and recorded with good driving behavior, which emphasizes staying in the center of the lane as much as possible. The following is an example image of center lane driving:

![center][image2]


Samples involving recoveries to the center lane included a several sequences throughout the lap that may look like the following sequence:


![sway1][image3]
![sway2][image4]
![sway3][image5]
![sway4][image6]

Starting from the top, 1. the car is heading towards the right-side edge of the road, 2. steering is applied towards the center, 3. the car has overshot a bit too much to the left, and 4. eventually readjust back into the center of the lane. The idea is to record samples of such recovery back to the center, so that the network will be able to do the same in similar situations.

Flipped versions of the collected images were augmented to the data set. Since the car is oriented counter-clockwise in the track, earlier models had a bias towards turning left. To deal with this left turn bias, flipping images and taking the opposite sign of the steering measurement provided examples for right turn bias, which helps balance out the bias and also generalize the model. Below is an example image taken from the center camera, followed by it's flipped version.

![pre][image7]

![post][image8]


After this process, there were 93,240 data points available. Since the dataset was very large, a generator was used to pass the images in batches, reducing the working set size. This approach is memory efficient and helped improve the training speed. Shuffling the data was done as part of the generator. The training and validation data split was done at a ratio of 8:2. 

The image pre-processing was 'built-in' to the network. The input images were normalized, and then cropped to remove the areas of the image containing sky and the body of the car, prior to the first convolution layer of the network.

