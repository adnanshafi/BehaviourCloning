import random
import csv
from sklearn.utils import shuffle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Dropout
from keras.layers.convolutional import Conv2D
from keras.regularizers import l2
import tensorflow as tf


# Read the collected Data
# Make sure the training data is inside a folder calld 'data'
# The csv file is in './data/csvfile.csv'
# The training images are in './data/IMG' folder

data_with_headers = []
with open('./data/driving_log.csv', 'r') as csvfile:
    #  Read the csv file
    reader = csv.reader(csvfile)
    for line in reader:
        data_with_headers.append(line)
data_without_headers = data_with_headers[1:]
data = []
# Probabilty for keeping images with steering angles == 0 
keep_prob = 0.3 
for line in data_without_headers:
    if abs(float(line[3])) == 0:
        # Don't much include data with steering angle == 0
        if np.random.uniform(0.0, 1.0) < keep_prob:
            data.append(line)
    else :
        data.append(line)


# Helper functions to augment the images
def add_random_shadow(image, y):
    """
    Helper function to add random shadow to the image.
	https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
    Args :
        image: The input image
        y : The steering angle
    Returns : 
        Tuple of processed image and steering angle 
    """ 
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2BGR)
    return image, y


# In[4]:

## Data augmentation helper functions : 
def rotate_random(src, y):
    """
    Rotate the image by a random small angle
    between -16 and 16 degrees.
    
    Args: src - Source image to be rotated
            y - Steering angle
    Returns: 
        Tuple of processed image and steering angle 
    """
    random_angle = np.random.randint(-16, 16)
    return rotate(src, random_angle, mode='nearest', reshape=False), y


def translate_random(src, y):
    """
    Translate in the given image by a random pixel value
    between 5 pixels in all four directions
    
    Args: 
        src - Source image to be translated
        y   - Steering angle
    Returns: 
        Tuple of processed image and steering angle 
    """
    rand_x = np.random.randint(-25,25)
    rand_y = np.random.randint(-25,25)
    translation_matrix = np.float32([ [1,0,rand_x], [0,1,rand_y]])
    return cv2.warpAffine(src, translation_matrix, (320, 160)), y
#     return src, y

def flip_horizantal(src, y):
    """
    Flip the image in horizantle direction

    Args : 
        src -  The input image to be flipped
        y   -  The Steering angle
    Returns:
        Tuple of processed image and steering angle 
    """
    flipped_img = cv2.flip(src, 1)
    return flipped_img, -1 * y


def random_brightness(src, y):
    """
    Increase or decrease the brightness of the image randomly

    Args : 
        src - The input image
        y   - The Steering angle
    Retunrns:
        Tuple of processed image and steering angle 
    """
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    rand = [60, 70, 80, -60, -70, -80]
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:,:,2] += random.choice(rand)
    hsv[:,:,2][hsv[:,:,2] > 255] = 255
    hsv[:,:,2][hsv[:,:,2] < 0] = 0
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), y

def augment_random(x, y):
    """
    Randomly chooses one of the functions to augment data and applies it 
    to the input image

    Args : 
        x - The input image to be augmented
        y - The steering angle
    Returns : 
        (x, y) - Augmented image and the steering angle

    """
    f = random.choice([lambda x, y : rotate_random(x, y),
                       lambda x, y : translate_random(x, y),
                       lambda x, y : flip_horizantal(x, y),
                       lambda x, y : random_brightness(x, y), 
                       lambda x, y : add_random_shadow(x, y),
                       lambda x, y : flip_horizantal(x, y)])
    return f(x, y)


# Shuffle the data and split into training and validation set 

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(data, test_size=0.2)
num_train_samples = len(train_samples)
num_valid_samples = len(validation_samples)


def generateMoreData(data, step_size, output_size, augmentations):
    """
        Generator to read the training images from the disk and 
        augment them and genrate training batches

        Args : 
            data        : Array of training data read from the csv file 
            step_size   : The number of lines to be read from csv files at a time
            output_size : The size of the batch outputter
            augmentations : Number of random augmentations to be applied
        Yields : 
            X_batch, y_batch : Features and labels 
    """
    
    data = shuffle(data)
    number_of_samples = len(data)
    # Steering correction for left and right images
    correction = 0.2
    while 1:
        for offset in range(0, int(number_of_samples), int(step_size)):
            batch_samples =  data[int(offset):int(offset+step_size)]
            images = []
            angles = []
            for sample in batch_samples:
                angle = float(sample[3])
                for i in range(3):
                    filepath = 'data/IMG/' + sample[i].split('/')[-1]
                    image = np.array(cv2.imread(filepath, 1))
                    if i == 1:
                        angle += correction
                    if i == 2:
                        angle -= correction
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    image = cv2.resize(image, (200, 160))
                    images.append(image)
                    angles.append(angle)                    
                    for i in range(augmentations):
                        aug_image, aug_angle = augment_random(image, angle)
                        aug_image = cv2.cvtColor(aug_image, cv2.COLOR_BGR2YUV)
                        aug_image = cv2.resize(aug_image, (200, 160))
                        images.append(aug_image)
                        angles.append(aug_angle)
            sample_size = len(images)
            images, angles = shuffle(images, angles)
            batches = 0
            for off in range(0, sample_size, output_size):
                batches += 1
                X_batch = np.array(images[off:off+output_size])
                y_batch = np.array(angles[off:off+output_size])
                yield shuffle(X_batch, y_batch)            


# Training params
# In case you run out of memory lower the multiplier size
dropout = 0.5
multiplier = 4
BATCH_SIZE = 128
STEP_SIZE = BATCH_SIZE * multiplier * 6
augmentations = 2
steps_train = (num_train_samples *  (augmentations + 1) * 3) // BATCH_SIZE
steps_valid = (num_valid_samples *  (1+1) * 3) // BATCH_SIZE
train_generator = generateMoreData(train_samples, STEP_SIZE, BATCH_SIZE, augmentations)
validation_generator = generateMoreData(validation_samples, STEP_SIZE, BATCH_SIZE, 1)
print ("Steps : training, validation >>", steps_train, steps_valid)
print ("Number of Training Images : ", steps_train * BACTH_SIZE)
print ("Number of Validation Images : ", steps_valid * BATCH_SIZE)


def makeModel(dropout):
    """
    The architecture of the Neural network used is defined here
    Args : 
        dropout : Keep probablity for dropping out
    Returns : 
        Keras model of the network
    """
    
    # activation used
    activation = 'relu'
    
    model = Sequential()
    
    # Cropping layer to  crop input image(160, 200, 3) to (66, 200, 3)
    model.add(Cropping2D(cropping=((69, 25), (0, 0)), input_shape=(160, 200, 3)))
    
    # Normalise the image with mean 0 and max abs value 0.5
    model.add(Lambda(lambda x: x/255 - 0.5))
    
    # Convolutional layer (66, 200, 3) to (31, 98, 24)
    model.add(Conv2D(kernel_size=(5,5),
                     strides=(2,2),
                     filters=24,
                     activation=activation,
                     padding='valid'))
    
    # Convolutional layer (31, 98, 24) to (14, 47, 36)
    model.add(Conv2D(kernel_size=(5,5),
                     strides=(2,2),
                     filters=36,
                     activation=activation,
                     padding='valid'))
    
    # Convolutional layer (14, 47, 36) to (5, 22, 48)
    model.add(Conv2D(kernel_size=(5,5),
                     strides=(2,2),
                     filters=48,
                     activation=activation,
                     padding='valid'))
    
    # Convolutional layer (5, 22, 48) to (3, 20, 64)
    model.add(Conv2D(kernel_size=(3,3),
                     strides=(1,1),
                     filters=64,
                     activation=activation,
                     padding='valid'))
    
    # Convolutional layer (3, 20, 64) to (1, 18, 64)
    model.add(Conv2D(kernel_size=(3,3),
                     strides=(1,1),
                     filters=64,
                     activation=activation,
                     padding='valid'))
    
    # Flattening (1, 18, 64) to 1152 neurons
    model.add(Flatten())
    
    # Fully Connected 1152 - 100
    model.add(Dense(100 , activation=activation))
    model.add(Dropout(dropout))
    
    # Fully Connected 100 - 50
    model.add(Dense(50 , activation=activation))
    model.add(Dropout(dropout))
    
    # Fully Connected 50 - 100
    model.add(Dense(10 , activation=activation))
    model.add(Dropout(dropout))
    
    # Steering output
    model.add(Dense(1))

    # compile the model
    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss='mse')
    return model


# Get the model
model = makeModel(dropout=dropout)


# Fit the model
history_object = model.fit_generator(generator=train_generator,
                                     steps_per_epoch=steps_train,
                                     validation_data=validation_generator,
                                     validation_steps=steps_valid,
                                     epochs=7,
                                     verbose=1)


# save the model
model.save('model.h5')



# Plot losses
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()