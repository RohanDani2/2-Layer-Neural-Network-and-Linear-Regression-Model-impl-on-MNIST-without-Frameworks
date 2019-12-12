# 2-Layer-Neural-Network-and-Linear-Regression-Model-impl-on-MNIST-without-Frameworks
Trained Neural Net and Lin Reg model to analyze accuracy on MNIST data set 

Linear Regression Model 

Abstract
	 In this case I had chosen the Linear Classifier. The Batch Training Error was 23% and the Test Error was 11.93% as this was the expected error. That’s good as the estimated error is roughly 12%. 
Theoretical Description 
	In a linear classifier model you can chose between different basis functions such as polynomials, Gaussian, splines, Logistic sigmoid, etc. The one used in the project is logistic sigmoid and for that is used when for the output there is a probability that has to be predicted since the range of it is 0 and 1. A linear classifier model is basically a one layer neural network where there is no hidden layer computed. One would want to choose a basis function based off the pros and cons of each for example the polynomial is sensitive to outliers and perturbations so one wouldn’t want to choose with a sporadic dataset. Essentially the linear regression model attempts to fit a linear equation to given data. For the single layer network one wants to separate each class from the other class then have a final score of all the classes with the sum of the outputs labeled i/x minus the sum of the outputs of the units labeled y/I for all of the x and y. For the linear classifier model for the mnist data, each input pixel contributes to a weighted sum for the output which then including the bias indicates the class of the input character. The benchmark for the linear classifier is at 12% whereas the test error can go to 8.4%. The error can be optimized with say regularized least squares. 
 
The Adam algorithm was implemented as that is first order gradient based optimization of functions. It’s an adaptive learning rate method as it uses the squared gradients to scale the learning rate by taking advantage of the momentum by using moving average of the gradient. Currently the hyperparameters that work best are eps = 1e^-8, beta1 = 0.9, and beta2 = 0.999. 
Implementation Details
	 	
For my implementation of the Logistic Sigmoid function was chosen. Softmax activation function and the derivative was used because it can be for multiclassification in the regression model, the probabilities sum will be 1, and the high value will have the higher probability than the other values. 
 
Onehot function was utilized to take the labels that were taken from the mnist data and onehot encode them that way they were separated by their class labels. This aids in the multi class classification because we’re dealing with huge dataset. 
 
Here is where the Adam algorithm was set as the gradient that was calculated earlier is used to determine the weights iteratively. 
 
Reproducibility
To implement this in Julia, you’ll need to have the HDF5, Distributions, Linear Algebra, and Random function implemented. The hyperparameters will have to be experimented on but ideally as long as they aren’t that farfetched it should still converge. The Adam algorithm formula with its parameters will have to be set and those include the eps, beta1, and beta2. The beta1 and beta2 will be needed as decay rates for momentum and the eps is a numerical stability parameter to avoid divide by 0 issue. 
Results 
This results I received for the final Test error was 0.1196 so 11.96%. It took 76.42108 seconds, 87.55 M allocations, 66.935 GiB, and 9.84% gc time. For a data set this extensive that’s pretty efficient however the tradeoff is the accuracy as some other models have far lower benchmarks. 

2 Layer Neural Network

Abstract
I had decided to implement the 2 layer neural network and the error I received for the Batch Training error is 13.95% and the Batch Test Error is 
Theoretical Description 
A neural network is essentially a combination of nodes which are grouped into layers that are inputted into a linear model and the two layer is of weights with one hidden layer. Stochastic backpropagation can be used to minimize the error function. For a two layer network it follows this method below to determine the outputs of the layers. During the movement of the network, as the learning proceeds the weights grow which increases the capacity of the network. The ReLU activation function could’ve been used as this has a range from 0 to infinity as the function and derivative are both monotonic. However, the issue is that the negative values turn 0 which hurts ability for the model to fit or train the data properly. 
  
Implementation Details
   

For the 2 layer network this is the softmax and softmax derivative function application instead of the sigmoid function for the multiclass classification. 
 
Here the two gradients are being calculated and I was able to reduce the number of for loops through the use of dot products which speeds up the algorithm drastically. The back propagation is done by multiplying the weight value by the delta and adding it to the gradient vector. 
 
The Adam algorithm is computed as the gradients are divided by the batch size as well and all the scalars were dot multiplied to the matrices to make sure that the right value was being calculated. 

Reproducibility
To reproduce this 2 layer network the packages I used were Plots, Images, Distributions, Linear Algebra, HDF5, Random, Special Functions, and StatsPlots. To reproduce what I had implemented there will be a lot of changing of hyperparameters and reducing the number of loops used as you can see the difference from the commented code to the one not.  
 
Results 
	This result I received was 7.58% testing error which is good as the benchmark is roughly around 7.6%. The time taken for computing was about 1 hour and 13 minutes which is a long time but it did reduce the error by a lot from the 1 layer. 
References 
http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c#:~:targetText=Adam%20%5B1%5D%20is%20an%20adaptive,for%20training%20deep%20neural%20networks.&targetText=The%20algorithms%20leverages%20the%20power,learning%20rates%20for%20each%20parameter.
http://cs231n.github.io/neural-networks-3/?fbclid=IwAR35GY8AFPVyudC_Ypf7TNPBq7Ie7MyrVTU5384YGKNQhYexKeBEgmTjoLI
https://medium.com/aidevnepal/for-sigmoid-funcion-f7a5da78fec2
https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

