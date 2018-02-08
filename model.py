# model.py

# Read in CSV data
import csv

#data_path = './data'				# directory containing sample track1 data
#data_path = './p3_train_data/'		# data collected from simulation
data_path = './data_combined/'		# both data included

lines = []
with open(data_path + 'driving_log.csv', 'r') as csvfile:
	#csvfile.readline() # skip header, needed for provided data
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# Set up train/validation split at 8:2 ratio
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

import cv2
import numpy as np
import sklearn


## Define generator for processing images on-the fly
## This helps performance by being memory efficient
def generator(samples, batch_size=32):
	
	num_samples = len(samples)
	while 1:

		# Shuffle image order
		sklearn.utils.shuffle(samples)

		# Break up the samples into batches 
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			# Load in images and label for center, left, right cameras
			images = []
			measurements = []
			for line in batch_samples:

				steering_center = float(line[3])
	
				# Create adjusted steering measurements for side camera images
				correction = 0.15
				steering_left = steering_center + correction
				steering_right = steering_center - correction

				# Read in center, left and right images
				path = data_path + 'IMG/'
				src_center = path + line[0].split('/')[-1]
				src_left = path + line[1].split('/')[-1]
				src_right = path + line[2].split('/')[-1]

				img_center = cv2.imread(src_center)
				img_left = cv2.imread(src_left)
				img_right = cv2.imread(src_right)

				images.extend((img_center, img_left, img_right))
				measurements.extend((steering_center, steering_left, steering_right))

			# Augment flipped images and measurements
			aug_images, aug_measurements = [], []
			for image, measurement in zip(images, measurements):
				aug_images.append(image)
				aug_measurements.append(measurement)
				aug_images.append(cv2.flip(image, 1))
				aug_measurements.append(measurement * -1.0)

			# Convert to numpy array
			X_train = np.array(aug_images)
			y_train = np.array(aug_measurements)
			yield sklearn.utils.shuffle(X_train, y_train)


# Define generators for training and validation
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)



# Hyperparameters
EPOCHS = 10
DROP_RATE = 0.5


## Model Architecture
## Nvidia CNN with Dropout
## Based on model presented in http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

# Import necessary classes
from keras.models import Sequential
from keras.layers import Activation, Cropping2D, Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()

# Normalize Input 
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

# Cropping layer - remove parts of sky/environment and car body from images
# Removes top 70 rows and bottom 25 rows, and 0 columns
model.add(Cropping2D(cropping=((70,25), (0,0))))

# First 3 conv layers use 5x5 kernels with stride of 2x2
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))

# Last 2 conv layers use 3x3 kernels with stride of 1x1
model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation="relu"))
model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation="relu"))

# Fully Connected layer, with dropout for regularization
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dropout(DROP_RATE))
model.add(Dense(50, activation="relu"))
model.add(Dropout(DROP_RATE))
model.add(Dense(10))
model.add(Dense(1))



### Model training and evaluation

# Minimize error between predicted and actual steering values
# The Adam optimzer is used
model.compile(loss='mse', optimizer='adam')

# Add early stopping to avoid overfitting to training data
early_stop = EarlyStopping(monitor='val_loss', min_delta = 1e-5, patience=3, verbose=1) 

# Add checkpointing to save model every epoch if it is an improvement
checkpoint = ModelCheckpoint('model_cp.h5', monitor='val_loss', save_best_only=True, verbose=0)

# Feed batches from train and validation data generators
# This is more memory efficient, improves performance by reducing working set size 
model.fit_generator(train_generator, samples_per_epoch = len(train_samples) * 3 * 2, validation_data=validation_generator, nb_epoch=EPOCHS, nb_val_samples = len(train_samples) * 3 * 2, callbacks=[early_stop, checkpoint])

# Save model
model.save('model.h5')
