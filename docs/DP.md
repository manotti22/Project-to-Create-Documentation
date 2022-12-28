# Project information


# Deep Learning
The result of machine learning program above section to classify genre of movie poster is good enough for simple cases. but above program use handcrafted features that not always fit with certain large-scale data. For example we want to make an output as similar as an input (image generation). Given x = {images-1,..images-n}, we want to draw output y as similar as x, this case can be solved using Auto-Encoder (Read my previous articles). another case is imagine we want to classify genre movie poster without handcrafted feature(edge orientation of image, RGB Histogram and so on) and change it into Neural Network that can make result as similar as data. In this project, we want to classify genre movie poster using Deep Neural Network with step as follows:

Downloading Dataset
Defining a Model
Training Model
Evaluating Model
Testing Model

# Program 1
```
Deep learning can be developed by using several tools or libraries like Tensorflow, Pytorch and so on. in this tutorials, we will use Tensorflow running on Python. The first step is to install environment tools like Anaconda to easily developed Python code and its libraries. Python is already available in Anaconda, so you dont have to install it anymore. Tensorflow should be install after finishing Anaconda installation by following this Tensorflow Installation in Conda. We will create simple Neural Network (Perceptron) as follows:

Fig.2

There are two input: x1 = 1 and x2 = 0, with initialization weight w1 = -0.5 and initialization w2 = 0.2. We want to compute y (output) as similar as ground truth / y_true so we need to arrange our architecture well. input or output can't be changed but we can modify value of w1 and w2 in order to make input as similar as output. let say we have x = image of cat, y1 = cat label and y2 = dog label. so our system should be make x as similar y1.

In this case, we give y_true = 1. Before computing y, we have to compute h1 first. output from h1 will be activated using sigmoid function. Later, we will use deep learning architecture that consists of more hidden like h1 to produce y as similar as y_true. Here step-by-step perceptron implementation in Tensorflow:

Import Tensorflow library.
import tensorflow as tf
in the code above, tensorflow is aliased by tf. so later you just called 'tf' to use tensorflow

Define input data and output data as constant.
x1 = tf.constant(1.0,name='x1')
x2 = tf.constant(0.0,name='x2')
y_true = tf.constant(1.0,name='y_true')
in the code above, tf.constant can be used to define constant value and store it into x1, x2 and y_true.

define weight
w1 = tf.Variable(-0.5,name='w1')
w2 = tf.Variable(0.2,name='w2')
in the code above, tf.Variable can be used to define variable (can be modify) and give default value -0.5 and 0.2 respectively to w1 and w2.

Define hidden layer.
h11 = tf.multiply(x1,w1)
h12 = tf.multiply(x2,w2)
h1 = tf.add(h11,h12)
in the code above, we do multiplication between x1 and w1 (look the image architecture) and multiplication between x2 and w2. the result of both multiplication will be added into h1.

Define output layer
y_predict = tf.nn.sigmoid(h1)
in the code above, we give activation function(sigmoid) to h1. you can check Tensorflow nn library for more activationa function. The result is stored at 'out' variable.

define loss function
loss = tf.pow((y_predict - y_true),2)
in the code above, we define loss function using Mean Square Error (MSE). we want to know how much loss of y_predict(computation in your model) to ground truth (y_true).

Fig.3 MSE from Wikipedia

define optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
in the code above, we define Gradient Descent Optimizer to compute gradient of architecture (including weight) so we can do backpropagation to update weight. Learning rate is the step of gradient descent to reach optimum. you can set learning_rate higher to quicken reach global optimum, but be careful of trapping of local optimum

compute Gradient
grad = optimizer.compute_gradients(loss)
in the code above, gradient descent will be worked to compute all of gradient from loss until input. is that finish? NO, you have not run the model. The model have just created, but you have not run the model.

define a session
sess = tf.Session()
in the code above, tf.Session() can be useful to define session of program run. to run a program, just called sess.run(..)

running all variables
sess.run(tf.initialize_all_variables())
you have already defined x1,x2,w1,w2,h1,y,loss that should be run first in order to compute a gradient. to run initialize variables that already define, use tf.initialize_all_variables()

running apply gradient
sess.run(optimizer.apply_gradients(grad))
print(sess.run(y_predict))
print(sess.run(w1))
print(sess.run(w2))
#0.377541
#-0.5
#0.2
after apply gradient to optimizer, calculation of y_predict = 0.377541 with w1 = -0.5 (default w1) and w2 = 0.2 (default w2). that result still far away from y_true = 1. so we need more computation of gradient and update weight of networks!

Computation and update weight of networks as epochs
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
for i in range(10):
  sess.run(train_step)
  print('epoch ',str(i),' : ',str(sess.run(y_predict)))
#epoch 0 : 0.379261
#...
#epoch 9 : 0.394791
in the code above, we try to do 10 epochs(the num of compute gradient and update weight of architecture) we can see the result of y_predict = 0.394791 in epoch 9 after computing gradient and updating gradient. The result is too far away from y_true = 1. so we can increase learning_rate = 0.5

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
for i in range(10):
  sess.run(train_step)
  print('epoch ',str(i),' : ',str(sess.run(y_predict)))
#epoch 0: 0.463863
..
#epoch 9: 0.0684392
still far away from y_true, so training in more epochs

for i in range(1000):
  sess.run(train_step)
  print('epoch ',str(i),' : ',str(sess.run(y_predict)))
#epoch 0: 0.684392
..
#epoch 999: 0.980032
in the code above, we try to do 1000 epochs to compute gradient and update the weight with the result is 0.980032. you can do update weight and compute gradient until convergence (no update again/value can not be updated)
```
# Program 2
```
We are going deeper with Tensorflow, Let say i have Multi Layer Perceptron like below picture:
Fig.3

There are several layer: 2 input layer[1,0], hidden1 = 32 neuron, hidden2 = 32 neuron and output layer (y_true) = 1. we want to initialize weight and bias randomly. each output from hidden layer will be activated using ReLU and in the output layer will be activated using Sigmoid. our task is to training a data so y_predict as similar as y_true. Here step-by-step perceptron implementation in Tensorflow:

Convolutional Neural Network
```

