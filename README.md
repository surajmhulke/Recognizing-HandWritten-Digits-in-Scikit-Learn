# Recognizing-HandWritten-Digits-in-Scikit-Learn

Scikit learn is one of the most widely used machine learning libraries in the machine learning community the reason behind that is the ease of code and availability of approximately all functionalities which a machine learning developer will need to build a machine learning model. In this article, we will learn how can we use sklearn to train an MLP model on the handwritten digits dataset. Some of the other benefits are:

It provides classification, regression, and clustering algorithms such as the SVM algorithm, random forests, gradient boosting, and k-means.
It is also designed to operate with Python’s scientific and numerical libraries NumPy and SciPy. Scikit-learn is a NumFOCUS project that has financial support.
Importing Libraries and Dataset
Let us begin by importing the model’s required libraries and loading the dataset digits.

# importing the hand written digit dataset
from sklearn import datasets
 
# digit contain the dataset
digits = datasets.load_digits()
 
# dir function use to display the attributes of the dataset
dir(digits)
Output:

['DESCR', 'data', 'feature_names', 'frame', 'images', 'target', 'target_names']
Function to print a set of image
digits.image is a three-dimensional array. The first dimension indexes images, and we can see that there are 1797 in total. 
The following two dimensions relate to each image’s pixels’ x and y coordinates. 
Each image is 8×8 = 64 pixels in size. In other terms, this array may be represented in 3D as a stack of 8×8 pixel images. 
# outputting the picture value as a series of numbers
print(digits.images[0])
Output:


[[ 0.  0.  5. 13.  9.  1.  0.  0.]
 [ 0.  0. 13. 15. 10. 15.  5.  0.]
 [ 0.  3. 15.  2.  0. 11.  8.  0.]
 [ 0.  4. 12.  0.  0.  8.  8.  0.]
 [ 0.  5.  8.  0.  0.  9.  8.  0.]
 [ 0.  4. 11.  0.  1. 12.  7.  0.]
 [ 0.  2. 14.  5. 10. 12.  0.  0.]
 [ 0.  0.  6. 13. 10.  0.  0.  0.]]
The original digits had much higher resolution, and the resolution was reduced when preparing the dataset for scikit-learn to allow training a machine learning system to recognize these digits easier and faster. Because at such a low resolution, even a human would struggle to recognize some of the digits The low quality of the input photos will also limit our neural network in these settings. Is the neural network capable of doing at least as well as an individual? It would already be an accomplishment!

# importing the matplotlib libraries pyplot function
import matplotlib.pyplot as plt
# defining the function plot_multi
 
def plot_multi(i):
    nplots = 16
    fig = plt.figure(figsize=(15, 15))
    for j in range(nplots):
        plt.subplot(4, 4, j+1)
        plt.imshow(digits.images[i+j], cmap='binary')
        plt.title(digits.target[i+j])
        plt.axis('off')
    # printing the each digits in the dataset.
    plt.show()
 
    plot_multi(0)
Output:
![image](https://github.com/surajmhulke/Recognizing-HandWritten-Digits-in-Scikit-Learn/assets/136318267/2bddba46-87e0-49d2-8912-c27febec1726)

Batch of Hand Written Digit Dataset
A batch of 16 Hand Written Digits

Training Neural network with the dataset
A neural network is a set of algorithms that attempts to recognize underlying relationships in a batch of data using a technique similar to how the human brain works. In this context, neural networks are systems of neurons that might be organic or artificial in nature.

an input layer consisting of 64 nodes, one for each pixel in the input pictures They simply send their input value to the neurons of the following layer.
This is a dense neural network, meaning each node in each layer is linked to all nodes in the preceding and following levels.
The input layer expects a one-dimensional array, whereas the image datasets are two-dimensional. As a result, flattening all images process takes place:

# converting the 2 dimensional array to one dimensional array
y = digits.target
x = digits.images.reshape((len(digits.images), -1))
 
# gives the  shape of the data
x.shape
Output:

(1797, 64)
The 8×8 images’ two dimensions have been merged into a single dimension by composing the rows of 8 pixels one after the other. The first image, which we discussed before, is now represented as a 1-D array having 8×8 = 64 slots.

# printing the one-dimensional array's values
x[0]
Output:

array([ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.,  0.,  0., 13., 15., 10.,
       15.,  5.,  0.,  0.,  3., 15.,  2.,  0., 11.,  8.,  0.,  0.,  4.,
       12.,  0.,  0.,  8.,  8.,  0.,  0.,  5.,  8.,  0.,  0.,  9.,  8.,
        0.,  0.,  4., 11.,  0.,  1., 12.,  7.,  0.,  0.,  2., 14.,  5.,
       10., 12.,  0.,  0.,  0.,  0.,  6., 13., 10.,  0.,  0.,  0.])
Data split for training and testing.
When machine learning algorithms are used to make predictions based on data that was not used to train the model, the train-test split process is used to measure their performance.

It is a quick and simple technique that allows you to compare the performance of algorithms for machine learning for your predictive modeling challenge.

# Very first 1000 photographs and
# labels will be used in training.
x_train = x[:1000]
y_train = y[:1000]
 
# The leftover dataset will be utilised to
# test the network's performance later on.
x_test = x[1000:]
y_test = y[1000:]
Usage of Multi-Layer Perceptron classifier
MLP stands for multi-layer perceptron. It consists of densely connected layers that translate any input dimension to the required dimension. A multi-layer perception is a neural network with multiple layers. To build a neural network, we connect neurons so that their outputs become the inputs of other neurons.

# importing the MLP classifier from sklearn
from sklearn.neural_network import MLPClassifier
 
# calling the MLP classifier with specific parameters
mlp = MLPClassifier(hidden_layer_sizes=(15,),
                    activation='logistic',
                    alpha=1e-4, solver='sgd',
                    tol=1e-4, random_state=1,
                    learning_rate_init=.1,
                    verbose=True)
Now is the time to train our MLP model on the training data.

mlp.fit(x_train, y_train)
Output:

Iteration 185, loss = 0.01147629
Iteration 186, loss = 0.01142365
Iteration 187, loss = 0.01136608
Iteration 188, loss = 0.01128053
Iteration 189, loss = 0.01128869
Training loss did not improve more than tol=0.000100 for 10 consecutive epochs.
Stopping.
MLPClassifier(activation='logistic', hidden_layer_sizes=(15,),
              learning_rate_init=0.1, random_state=1, solver='sgd',
              verbose=True)
Above shown is the loss for the last five epochs of the MLPClassifier and its respective configuration.

fig, axes = plt.subplots(1, 1)
axes.plot(mlp.loss_curve_, 'o-')
axes.set_xlabel("number of iteration")
axes.set_ylabel("loss")
plt.show()
Output:

![image](https://github.com/surajmhulke/Recognizing-HandWritten-Digits-in-Scikit-Learn/assets/136318267/27722ba5-fa65-4107-a475-e9da6ec6e09a)

 

Model Evaluation
Now let’s check the performance of the model using the recognition dataset or it just has memorized it. We will do this by using the leftover testing data so, that we can check whether the model has learned the actual pattern in the digit 

predictions = mlp.predict(x_test)
predictions[:50]
Output:

array([1, 4, 0, 5, 3, 6, 9, 6, 1, 7, 5, 4, 4, 7, 2, 8, 2, 2, 5, 7, 9, 5,
       4, 4, 9, 0, 8, 9, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 3, 0, 1, 2, 3, 4,
       5, 6, 7, 8, 5, 0])
But the true labels or we can say that the ground truth labels were as shown below.

y_test[:50]
Output:

array([1, 4, 0, 5, 3, 6, 9, 6, 1, 7, 5, 4, 4, 7, 2, 8, 2, 2, 5, 7, 9, 5,
       4, 4, 9, 0, 8, 9, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
       5, 6, 7, 8, 9, 0])
So, by using the predicted labels and the ground truth labels we can find the accuracy of our model.

# importing the accuracy_score from the sklearn
from sklearn.metrics import accuracy_score
 
# calculating the accuracy with y_test and predictions
accuracy_score(y_test, predictions)
Output:

0.9146800501882058
