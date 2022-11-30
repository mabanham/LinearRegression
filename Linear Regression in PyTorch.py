import numpy as np
import math
import jovian
import PIL
import torch as pt
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms as transforms

from torchvision.datasets import MNIST

# Download Training Data
dataset = MNIST(root='data/', download=True)

# When this statement is executed the first time, it downloads the data to the data/ directory next to the project and
# creates a PyTorch 'Dataset'. On subsequent executions, the download is skipped as the data is already downloaded.
# Check the size of the dataset:

print(dataset)
print(len(dataset))
# The data set has 60,000 images which can be used to train the model. There is also an additional test set of 10,000
# images which can be created by passing train=False to the MNIST class.

test_dataset = MNIST(root='data/', train=False)
print(test_dataset)
print(len(test_dataset))

# Let's look at a sample element from the training dataset.
print(dataset[0])

# It's a pair, consisting of a 28x28 image and a label. The image is an object of the class PIL.Image.Image, which
# is a part of the Python imaging library 'Pillow'. We can view the image using 'matplotlib'(see imports),
# the de-facto plotting and graphing library for data science in Python.

image, label = dataset[0]
plt.imshow(image, cmap='gray')
plt.show()
print('Label:', label)

image2, label = dataset[10]
plt.imshow(image2, cmap='gray')
plt.show()
print('Label:', label)

# This displays the first image and 11th image of the dataset in a separate window.
# It's evident that these are quite small images, and recognizing the digits can sometimes be hard even for a human
# While it is useful to look at these images, the problem here is that Pytorch doesn't know how to work with images.
# We need to convert the images into tensors. We can do this by specifying a transform while creating our dataset.
# import torchvision.transforms as transforms
# PyTorch datasets allow us to specify one or more transformation functions which are applied to the images as they are
# loaded. 'torchvision.transforms' contains many such predefined functions, and we'll use the 'ToTensor' transform to
# convert images into PyTorch tensors.

# MNIST dataset (images and labels)
dataset = MNIST(root='data/',
                train=True,
                transform=transforms.ToTensor())

img_tensor, label = dataset[0]
print(img_tensor.shape, label)

# This image is now converted to a 1x28x28 tensor. The first dimension is used to keep track of the colour channels.
# Since images in the MNIST dataset are grayscale, there's just one channel. Other datasets have images with colour,
# in which case there are 3 channels: RGB.
# Let's look at some sample values inside the tensor:

print('sample values from inside the tensor')
print(img_tensor[:, 10:15, 10:15])
print(pt.max(img_tensor), pt.min(img_tensor))

# The values range from 0 to 1, with 0 representing black, 1 white and the values in between different shade of grey.
# We can also plot the tensor as an image using plt.imshow.

# Plot the image by passing in the 28x28 matrix
plt.imshow(img_tensor[0, 10:15, 10:15], cmap='gray')
plt.show()

# Note that we need to pass just the 28x28matrix to 'plt.imshow', without a channel dimension. We can also pass a colour
# map 'cmap=gray' to indicate that we want to see a grayscale image.

# While building real world machine learning models, it is quite common to split the dataset into 3 parts:
#   1. Training sets-  Used to train the models i.e. compute the loss and adjust the weights of the model using
#                      gradient decent.
#   2. Validation set- Used to evaluate the model while training, adjust hyper-parameters (learning rate etc).
#                      and pick the best version of the model.
#   3. Test set-       Used to compare different models, or different types of modeling approaches, and report
#                      the final accuracy of the model.

# In the MNIST dataset there are 60,000 training images, and 10,000 test images. The test set is standardized so that
# different researchers can report the results of their models against the same set of images. Since there's no
# predefined validation set, we must manually split the 60,000 images into training and validation datasets.

# Let's define a function that randomly picks a given fraction of the images for the validation set.
# import numpy as np
'''
n = 60000
val_pct = 0.1 #10%
n_val = int(val_pct*n)
print(n_val)
'''


def split_indices(n, val_pct):
    # Determine size of validation set:
    n_val = int(val_pct * n)
    # Create random permutation of 0 to n-1
    idxs = np.random.permutation(n)
    # Pick first n_val indices for validation set
    return idxs[n_val:], idxs[:n_val]


# 'split_indices' randomly shuffles the array indices 0, 1, ...n-1, and separates out a desired portion from it for
# the validation set. It's important to shuffle the indices before creating a validation set, because the training
# images are often ordered by the target labels i.e. images of 0s, followed by images of 1s, etc...
# if we were to pick a 20% validation set simply by selecting the last 20% of images, the validation set would
# mostly consist of 8s and 9s and the training data would contain no 8s and 9s. Making a good model impossible
# to achieve.

train_indices, val_indices = split_indices(len(dataset), val_pct=0.2)
print(len(train_indices), len(val_indices))
print('sample val indices', val_indices[:20])

# We have randomly shuffled the indices, and selected a small portion (20%) to serve as the validation set. We can
# now create PyTorch data loaders for each of these using a 'SubsetRandomSampler', which samples elements randomly
# from a given list of indices, while creating batches of data.

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader

batch_size = 100

# Training sampler and data loader
train_sampler = SubsetRandomSampler(train_indices)
train_loader = DataLoader(dataset,
                          batch_size,
                          sampler=train_sampler)

# Validation sampler and data loader
val_sampler = SubsetRandomSampler(val_indices)
val_loader = DataLoader(dataset,
                        batch_size,
                        sampler=val_sampler)

# Model
# Now that we have prepared our data loaders, we can define our model.
#   -   A logistic regression model is almost identical to a linear regression model i.e. there are weights and bias
#       matrices, and the output is obtained using simple matrix operations 'pred = x @ w.t() + b'.
#   -   Just as we did with linear regression, we can use nn.Linear to create the model instead of defining and
#       initializing the matrices manually.
#   -   Since nn.Linear expects each training example to be a vector, each 1x28x28 image tensor needs to be flattened
#       out into a vector of size 784 (28*28), before being passed into the model.
#   -   The output for each image is vector of size 10, with each element of the vector signifying the probability
#       a particular target label (i.e. 0 to 9). The predicted label for an image is simply the one with the
#       highest probability
# import torch.nn as nn

input_size = 28*28
num_classes = 10

#Logistic regression model
model = nn.Linear(input_size, num_classes)

#Of course, this model is a lot larger than the previous model, in terms of the number of parameters.
# Let's take a look at the weights and biases.

print('model weights')
print(model.weight.shape)
print(model.weight)
print('model biases')
print(model.bias.shape)
print(model.bias)

# Although there are a total of 7850 parameters here, conceptually nothing has changed so far. Let's try and generate
# some outputs using our model. We'll take the first batch of 100 images from our dataset, and pass them into our model.
'''
for images, labels in train_loader:
    print(labels)
    print(images.shape)
    outputs = model(images)
    break
'''
# This leads to an error, because our input data does not have the right shape. Our images are of the shape 1x28x28,
# but we need them to be vectors of 784 i.e. we need to flatten them out. We'll use the .reshape method of a tensor,
# which allows us to efficiently 'view' each image as a flat vector, without really changing the underlying data.

# To include this additional functionality within our model, we need to define a custom model, by extending
# the nn.module class from Py.Torch. see 2:08:00 of: https://www.youtube.com/watch?v=GIsg-ZUy0MY&t=4283s

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        # PyTorch can  work out what the other dimension but putting a '-1' here generalises the model allowing us
        # to work with different batch sizes.
        out = self.linear(xb)
        return out

model = MnistModel()


# Inside the '__init__' constructor method, we instantiate the weights and biases using 'nn.Linear'. And inside the
# forward method, which is invoked when we pass a batch of inputs to the model, we flatten out the input tensor,
# and then pass it into 'self.linear'.
# 'xb.reshape(-1, 28*28) indicates to PyTorch that we want a view of the xb tensor with two dimensions, where the length
# along the 2nd dimension is 28*28(i.e.784). One argument to '.reshape' can be set to a -1 (in this case the first
# dimension), to let PyTorch figure it out automatically based on the shape of the original tensor.

# Note that the model no longer has '.weight' and '.bias' attributes (as they are now inside the '.linear' attribute),
# but it does have a '.parameters' method which returns a list containing the weights and biases, and can be used by
# PyTorch optimizer.

print(model.linear.weight.shape, model.linear.bias.shape)
print(list(model.parameters()))

# Our new custom model can be used in the exact same way as before. Let's see if it works:
for images, labels in train_loader:
    outputs = model(images)
    break

print('outputs.shape:', outputs.shape)
print('sample outputs:\n', outputs[:2].data)

# For each of the 100 input images, we get 10 outputs, one for each class. AS discussed earlier, we'd like these outputs
# to represent probabilities, but for that the elements of each output row must lie between 0 to 1 and add up to 1,
# which is clearly not the case here.

# Softmax(see 2:14:48 of : https://youtu.be/GIsg-ZUy0MY?t=8088)
# While it's easy to implement the softmax function, we'll use the implementation that's provided within PyTorch,
# because it works well with multidimensional tensors (a list of output rows in our case).

import torch.nn.functional as F

# the softmax function is included in the 'torch.nn.functional' package, and requires us to specify a dimension
# along which the softmax must be applied.

# Apply softmax for each output row
probs = F.softmax(outputs, dim=1)

# Look as sample probabilities
print("sample probabilities:\n", probs[2:].data)

# Add up the probabilities of an output row
print('Sum: ', pt.sum(probs[0]).item())

# Finally we can determine the predicted label for each image by simply choosing the index of the element with the
# highest probability in each output row. This is done using torch.max, (pt.max in my case), which returns the largest
# element and the index of the largest element along a particular diemnsion of a tensor.

max_probs, preds = pt.max(probs, dim=1)
print('prediction', preds)
print('Likelyhood:', max_probs)

# Let's compare this with the actual labels of the dataset.

print(labels)

# Obviously these predictions are completely different. That's because we have started with randomly initialized weights
# and biases. We need to train the model I.E. adjust the weights using gradient descent to make better predictions.


# Evaluation Metric and Loss Function

# Just as with linear regression, we need a way to evaluate how well our model is performing. A natural way to do this
# would be to find the percentage of labels that we predicted correctly I.E. the accuracy of the predictions.

def accuracy(preds, labels):
    return pt.sum(preds == labels).item() / len(labels)
# The == performs an element-wide comparison of two tensors with the same shape, and returns a tensor of the same shape,
# containing 0s for unequal elements, and 1s for equal elements (where the prediction is correct). Passing the result to
# torch.sum (pt.sum) returns the number of labels that were predicted correctly, Finally we divide that by the total
# number of images to get the accuracy
# Let's calculate the accuracy of the current model, on the first batch of data. Obviously, we expect it to be pretty
# bad.

print(accuracy(preds, labels))

# While the accuracy is a great way for us (humans) to evaluate the model, it can't be used as a loss function for
# optimizing our model using gradient descent, for the following reasons:
#   1.  It's not a differentiable fucntion. 'torch.max' and '==' are both non-continuous and non-differentiable
#       operations, so we can't use the accuracy for computing gradients with respect to the weights and biases.
#   2.  It doesn't take into account the actual possibilities predicted by the model, so it can't provide sufficient
#       feedback for incremental improvements.

# Due to these reasons, accuracy is a great evalutation metric for classification, but not a good loss function.
# A commonly used loss function for classification problems is the cross entropy, which has the following formula
# see: https://www.youtube.com/watch?v=GIsg-ZUy0MY&t=8363s

# For each output row, pick the predicted probability for the correct label. E.G. if the predicted probabilities for an
# image are [0.1, 0.3, 0.2, ...] and the correct label is 1, we pick the corresponding element and ignore the rest.
# We can think of these as vectors E.G.

[0, 0, 1, 0, 0, 0, 0, 0, 0, 0] #2 vector (if we multiply this by the following all the other probabilties go away)

[0.0682, 0.0862, 0.1345, 0.0978, 0.1111, 0.0864, 0.1039, 0.1038, 0.1108, 0.0972]
# We take the probability of #2 (0.1345)

# We then take the negative logarithm of the relevant probability.
print('negative log of relevant probability (example)')
print(-math.log(0.1345))

# Note that the lower the probability the higher the 'negative log function' returns because of the nature of the
# logarithm function as the probability goes up the negative log function decreases closer to 1.
# in this case we only care about how logarithm acts between 0 and 1.

# Finally, take the average of the cross entropy across all the output rows to get the overall loss for a batch of
# data.

# Unlike accuracy, cross-entropy is a continous and differentiable function that also provides good feedback for
# incremental improvements in the model (a slightly higher probability for the correct label leads to a lower loss).
# This makes it a good choice for the loss function.

# PyTorch provides an efficient and tensor-friendly implementation of cross entropy as part of the torch.nn.functional
# package. Moreover, it also performs softmax internally, so we can directly pass in the outputs of the model without
# converting them into probabilities.

loss_fn = F.cross_entropy

#Loss for the current batch

loss = loss_fn(outputs, labels)
print('loss')
print(loss)

# Since the cross entropy is the negative logarithm of the predicted probability of the correct label, averaged over all
# training samples; One way to interpret the resulting number e.g. 2.23 is look at e^-2.23 which is around 0.1 which is
# the predicted probability of the correct label, on average.

# Optimizer
# We are going to use the 'optim.SGD' optimizer to update the weights and biases during training, but with a higher
# learning rate of 1e-3.

learning_rate = 0.001
optimizer = pt.optim.SGD(model.parameters(), lr=learning_rate)

# Parameters like batch size, learning rate etc. need to be picked in advance while training machine learning models,
# and are called hyperparameters. Picking the hyperpapameters is critical for training an accurate model within a
# reasonable amount of time, and is an active area of research and experimentation. Try different learning rates and see
# how it effects the training process.

# Training the model

# Now that we have defined the data loaders, model, loss function and optimizer, we are ready to train the model.
# The training process is almost identical to linear regression. However, we'll augment the 'fit' function we defined
# earlier to evaluate the model's accuracy and loss using the validation set at the end of every epoch.

# We begin by defining a function loss_batch which:
#   Calculates the loss for a batch of data
#   Optionally perform the gradient descent up date step if the optimizer is provided
#   Optionally computes a metric (e.g. accuracy) using the prediction and actual targets

def loss_batch(model, loss_func, xb, yb, opt=None, metric=None):
    #Calculate loss
    preds = model(xb)
    loss = loss_func(preds, yb)

    if opt is not None:
        # Compute the gradients
        loss.backward()
        # Update parameters
        opt.step()
        # Reset gradients
        opt.zero_grad()

    metric_result = None
    if metric is not None:
        # Compute the metric
        metric_result = metric(preds, yb)

    return loss.item(), len(xb), metric_result

# The optimizer is an optional argument, to ensure that we can reuse loss_abth for computing the loss on the validation
# set. We also return the length of that batch as part of the result, as it'll be useful while combining the
# losses/metrics for the entire dataset.
# Validation set
def evaluate(model, loss_fn, valid_dl, metric=None):
    with pt.no_grad():
        # Pass each batch through the model
        results = [loss_batch(model, loss_fn, xb, yb, metric=metric)
    for xb, yb in valid_dl]
    # separate losses, counts and metrics
    losses, nums, metrics = zip(*results)
    # Total size of the dataset
    total = np.sum(nums)
    # Avg. loss across batches
    avg_loss =np.sum(np.multiply(losses,nums))/total
    avg_metric = None
    if metric is not None:
        # Avg. of metric across batches
        avg_metric = np.sum(np.multiply(metrics, nums)) / total
    return avg_loss, total, avg_metric

# If it's not immediately clear what this function does, try executing each statement seperately and see the results.
# We also need to redefine the accuracy to operate on an entire batch of outputs directly, so that we can use it as a
# metric in 'fit'.

def accuracy(outputs, labels):
    _, preds = pt.max(outputs, dim=1)
    return pt.sum(preds == labels).item() / len(preds)

# Note that we don't need to apply the softmax to the outputs, since it doesn't change the relative order of the
# results. This is because e^x is an increasing function i.e if y1 > y2, then e^y1 > e^y2 and the same holds true after
# averaging out the values to get the softmax.

# Let's see how the model performs on the validation set with the initial set of weights and biases.

val_loss, total, val_acc = evaluate(model, loss_fn, val_loader, metric=accuracy)
print('Loss: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_acc))

# The initial accuracy is around 10%, which is what one might expect from a randomly initialized model (since it has a 1
# in 10 chance of getting the label right by guessing randomly). Also note that we are using the '.format' method with
# the message string to print only the first four digits after the decimal point.

# We can now define the 'fit' function quite easily using loss_batch and evaluate. This takes the data batch by batch
# and runs gradient decent on it.

def fit (epochs, model, loss_fn, opt, train_dl, valid_dl, metric=None):
    for epoch in range (epochs):
        # Training
        for xb, yb in train_dl:
            loss, _, _ = loss_batch(model, loss_fn, xb, yb, opt)

        # Evaluation
        result = evaluate(model, loss_fn, valid_dl, metric)
        val_loss, total, val_metric = result

        # Print progress
        if metric is None:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, epochs, val_loss))

        else:
            print('Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f})'
                  .format(epoch+1, epochs, val_loss, metric.__name__, val_metric))

# We are now ready to train the model. Let's train for 5 epochs and look at the results.

# Redifine model and optimizer
model = MnistModel()
optimizer = pt.optim.SGD(model.parameters(), lr=learning_rate)

fit(100, model, F.cross_entropy, optimizer, train_loader, val_loader, accuracy)

# Excellent, after just 5 epochs of training, our model achieves an accuracy of over 80% on the validation set.
# Let's see if we can improve that by training it a bit more. (increase number of epochs to 100).
# While this does continue to increase accuracy as we train for more epochs, the improvements gets smaller with every
# epoch. This is easier to see in a line graph.

accuracies = [0.0945, 0.6014, 0.7175, 0.7596, 0.7850, 0.8007,
              0.8089, 0.8176, 0.8241, 0.8283, 0.8326, 0.8353,
              0.8386, 0.8411, 0.8442, 0.8469, 0.8494, 0.8515,
              0.8536, 0.8554, 0.8572, 0.8579, 0.8595, 0.8611,
              0.8625, 0.8644, 0.8658, 0.8678, 0.8681, 0.8685,
              0.8691, 0.8698, 0.8702, 0.8709, 0.8713, 0.8720,
              0.8727, 0.8738, 0.8745, 0.8756, 0.8767, 0.8772,
              0.8777, 0.8782, 0.8788, 0.8797, 0.8796, 0.8800,
              0.8804, 0.8806, 0.8809, 0.8816, 0.8819, 0.8826,
              0.8832, 0.8833, 0.8838, 0.8842, 0.8848, 0.8851,
              0.8857, 0.8858, 0.8863, 0.8868, 0.8868, 0.8874,
              0.8874, 0.8877, 0.8878, 0.8882, 0.8885, 0.8888,
              0.8894, 0.8898, 0.8903, 0.8903, 0.8908, 0.8908,
              0.8910, 0.8912, 0.8915, 0.8915, 0.8918, 0.8919,
              0.8920, 0.8921, 0.8922, 0.8922, 0.8929, 0.8928,
              0.8930, 0.8931, 0.8932, 0.8932, 0.8935, 0.8935,
              0.8937, 0.8938, 0.8941, 0.8943, 0.8948
              ]# First 100 Epochs
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs No. of epochs')
plt.show()

# It's clear from the graph that the model probably won't cross the accuracy threshold of 90% even after training for
# a very long time. One possible reason for this is that the learning rate might be too high. It's possible that the
# model's paramaters are "bouncing" around the optimal set of parameters that have the lowest loss. You can try reducing
# learning rate and training for a few more epochs to see if it helps.
#
# The more likely reason, is that the model just isn't powerful enough. Remember the initial hypothesis, we have assumed
# that the output (in this case the class probabilities) is a linear function of the input (pixel intensities), obtained
# by performing a matrix multiplication with the weights matrix and adding the bias. This is a fairly weak assumption,
# as there may not actually exist a linear relationship between the pixel intensities in an image and the the digit it
# represents. While it works reasonsably well for a simple dataset like MNIST (getting us to 85% accuracy), we need
# more sophisticated models that can capture non-linear relationships between image pixels and labels for complex tasks
# like recognizing everyday objects, animals etc.

# NOTE: We don't teach humans this way, with humans we understand how they are learning and adapt our teaching to be more
# efficient, example: if a child struggles to differentiate between 8 and 3, but has no problems with 1,2,4,5,6,7 and 9
# we don't keep asking them about 6's and 7's, but rather focus on the 8's and 3's. This is why it helps to identify
# where the model is failing in order to focus time on the areas it struggles.

# Testing with individual images
# While we have been tracking the overall accuracy of a model so far, it's also a good idea to look at model's results
# on some sample images. Let's test out the model with some images from the predefined test dataset of 10,000 images.
# Begin by recreating the test dataset with 'ToTensor' transform.

jovian.log_hyperparams({
    'opt': 'SGD',
    'lr': 0.001,
    'batch_size': 100,
    'arch': 'logistic-regression'
})

#Define test dataset
test_dataset = MNIST(root='data/',
                     train=False,
                     transform=transforms.ToTensor())

#Here's a sample image from the dataset.

img, label = test_dataset[0]
plt.imshow(img[0], cmap='gray')
print('Shape:', img.shape)
print('Label', label)

img.unsqueeze(0).shape
# this adds another dimension (1x28x28 becomes 1x1x28x28)
#Let's define a helper function predict_image, which return the predicted label for a single image tensor.

def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = pt.max(yb, dim=1)
    return preds[0].item()

# The model views this 4th dimension as a batch containing a single image

img, label = test_dataset[0]
plt.imshow(img[0], cmap='gray')
print('Label', label, 'Predicted:', predict_image(img, model))
plt.show()

img, label = test_dataset[193]
plt.imshow(img[0], cmap='gray')
print('Label', label, 'Predicted:', predict_image(img, model))
plt.show()

img, label = test_dataset[1198]
plt.imshow(img[0], cmap='gray')
print('Label', label, 'Predicted:', predict_image(img, model))
plt.show()

img, label = test_dataset[1742]
plt.imshow(img[0], cmap='gray')
print('Label', label, 'Predicted:', predict_image(img, model))
plt.show()

# Identifying where the model performs poorly can help us improve the model, by collecting more training data,
# increasing/decreasing the complexity of the model, and changing the hyperparameters.
# As a final step, let's also look at the overall loss and accuracy of the model on the test set.

test_loader = DataLoader(test_dataset, batch_size=200)

test_loss, total, test_acc = evaluate(model, loss_fn, test_loader, metric=accuracy)
print('Loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))



jovian.log_metrics({
    'val_loss': 0.3758,
    'val_acc': 0.9010
})
# We expect this to be similar to the accuracy/loss on the validation set. If not, we might need a better validation set
# that has similar data and distribution as the test set (which often comes from real world data).

# Saving and Loading the Model

# Since we trained our model for a long time and achieved a reasonable accuracy, it would be a good idea to save the
# weights and bias matrices to disk, so that we can reuse the model later and avoid retraining from scratch. Here's how
# to save the model.

pt.save(model.state_dict(), 'mnist-logistic.pth')

# The '.state_dict' method returns an 'OrderedDict' containing all the weights and bias matrices mapped to the right
# attributes of the model.

model.state_dict()

# To load the model weights, we can instante a new object of the class 'MnistModel', and use the '.load_state_dict'
# method.

model2 = MnistModel()
model2.load_state_dict(pt.load('mnist-logistic.pth'))
model2.state_dict()

# Just as a sanity check, let's verify that this mode has the same loss and accuracy on the test set as before.

test_loss, total, test_acc = evaluate(model, loss_fn, test_loader, metric=accuracy)
print('Loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))

# Save and commit
jovian.commit(outputs=['mnist-logistic.pth'])


