# Behaviorial Cloning Project

---

### Overview 

**Files included**  
  * model.py (script used to create and train the model)
  * drive.py (script to drive the car - feel free to modify this file)
  * model.h5 (a trained Keras model)
  * a report writeup file (either markdown or pdf)
  * video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)


### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)
* [Keras 2.0.2](https://anaconda.org/conda-forge/keras)
* Udacity Simulator
The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Steps to run the model:
1. Activate the carnd-term1 environment
2. Ensure you have Keras version 2.0.2. if not run 

```
conda install -c conda-forge keras=2.0.2
```
3. Run 
```
python drive.py model.h5
```
4. Open the Udacity Simulator and go to Autonomous mode 

### Steps to train the model: 
1. Collect data from the udacity simulater by driving the car around
2. Keep the data collected in a folder called data
3. Run model.py by typing 
```
python model.py
```

model.py expects to find images in `data/IMG` and csv file in `data/`

### Issues
In case while training the model uses Theano backend run with following command 
```
KERAS_BACKEND=tensorflow python model.py
KERAS_BACKEND=tensorflow drive.py model.h5
```
### The Model Architecture
[The model architecture can be found here](writeup.md)
