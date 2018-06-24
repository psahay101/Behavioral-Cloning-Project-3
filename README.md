# Behavioral Cloning project

[//]: # (Image References)

[image1]: ./model.png "Architecture "
[image2]: ./left.png "left camera"
[image3]: ./center.png "center camera"
[image4]: ./right.png "right camera"
[image5]: ./testdrive1.png "Test drive lane 1"

## Project Description
This project is about teaching a car to drive within a lane by training it on several images taken while a human drives a car. So, eventually the goal is to make the car emulate this behavior without just copying the behavior which in the technical terms can be called as overfitting.

To ensure that the car learns and not copies which will make it a fragile system, we implement many techniques which is the core of the project. Some of these techniques are implemented in the data pre-processing phase and the other techniques involve changing the neural network architecture.


## Files included
The following files will be included in this project:
* `model.py` : script used to create and train the model
* `drive.py` : script to drive the car.
* `model.h5` : a trained Keras model.
* a report writeup file (either markdown or pdf)
* `video.mp4` : a video recording of the vehicle driving autonomously around the track.

## Driving the car
The car can be driven manually by the user to record the data for training or else they can even import the data from the udacity github repository.

The driving is like any other PC game where you can use the arrow keys or AWSD to drive around and explore the amazing terrain.

After the training has been accomplished the model can be tested by letting the model take control of the game and this is done by using the autonomous mode in the game and simultaneously giving drive.py model.h5 command in the python in your environment.

## Model Architecture Design

The design of the network is based on the NVIDIA model, which has been used by NVIDIA for the end-to-end self driving test. As such, it is well suited for the project.
The model architecture is based on the NVIDIA model. There are a few changes like adding a dropout but mostly its the same. The reason that I decided to go for this exact network is because it has been tested and changed accordingly to produce one of the best architecture for this job.

In short this is a good model which can serve as a base model on top of which further changes can be implemented to further improve the results.

For my project, I changed the model by adding relu on each layer to introduce non linearity, added dropout to counter overfitting, normalized images to help in training by avoiding saturation.

Below is a snapshot of my architecture :

![][image1]

## Model Training
Before the training three major steps are carried out:
* Addition of left and right camera images into the main dataset.
* Randomization the dataset.
* changing brightness level of certain images randomly.
* Flipping the images laterally for some images randomly.

Photos of camera images adjusted for left and right are shown below:

![][image2]
![][image3]
![][image4]

For the training I used MSE of determining the loss and Adam optimizer with a learning rate of 0.001 for 10 epochs.

## Results

The Training loss and the validation loss were .
The car was able to drive smoothly on the designated track which was considered a success for the test.

![][image5]

The videos are found on Github repo 

## Future works
Further improvement scope is there and here are some ideas:
* Training for not just the steering angles but also for the throttle.
* Training for scenarios where car accidentally goes off the track into the rough terrain.
* Training in more kinds of lanes.
* Training to drive in one part of the lane.
