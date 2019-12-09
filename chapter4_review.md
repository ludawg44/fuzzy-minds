# Chapter 4 Training Models

- [Linear Regerssion](#linear-regression)
  - [The Normal Equation](#the-normal-equation)
  - [Computational Complexity](#computational-complexity)
- [Gradient Descent](#gradient-descent)
  - [Batch Gradient Descent](#batch-gradient-descent)
  - [Stochastic Gradient Descent](#stochastic-gradient-descent)
  - [Mini-batch Gradient Desecnt](#mini-batch-gradient-descent)
- [Polynomial Regression](#polynomial-regression)
- [Learning Curve](#learning-curve)
- [Regularized Linear Models](#regularized-linear-models)
  - [Ridge Regression](#ridge-regression)
  - [Lasso Regression](#lasso-regression)
  - [Elastic Net](#elastic-net)
  - [Early Stopping](#early-stopping)
- [Logistic Regression](#logistic-regression)
  - [Estimating Probabilities](#estimating-probabilities)
  - [Training and Cost Function](#training-and-cost-function)
  - [Decision Boundaries](#decision-boundaries)
  - [Softmax Regression](#softmax-regression)

There's a lot of math concepts in this chapter, specifically in Linear Algebra. Review this [linear algebra](https://github.com/ageron/handson-ml2/blob/master/math_linear_algebra.ipynb) repo Aurelien Geron created on GitHub.

## Linear Regression

This quick section goes over a linear regression model. You haven't trained anything yet. 

Key vocabulary: 
- bias term (intercept term)
- parameter vector
- feature vector
- column vectors
- Root Means Square Error (RMSE)

### The Normal Equation

Review the Normal Equalation. Review "from sklearn.linear_model import LinearRegression"

Key vocabulary: 
- Normal Equation
- closed-form solution
- pseudoinverse
- Singular Value Decomposition (SVD)

### Computational Complexity

Regarding the Linear Regression model, the computational complexity is linear with regard to both the number of instances you want to make predictions on and the number of features. 

Key vocabulary: 
- computational complexity

## Gradient Descent

The concept of a [gradient descent](https://www.pyimagesearch.com/2016/10/10/gradient-descent-with-python/) is incredibly important. But like in most ML problems, there are always faster and less computationally wasteful algorithms. Gradient descent is no different. 

This chapter mentions three modifications from the standard gradient descent: 1) batch gradient descent, stochastic gradient descent & mini-batch gradient descent. 

Key vocabulary: 
- gradient descent
- random initialization
- converges
- learning rate
- local minimum 
- global minimum
- convex function
- parameter space

### Batch Gradient Descent

Key vocabulary: 
- partial derivative 
- partial derivatives of the cost function
- gradient vector of the cost function
- batch gradient descent
- tolerance 

### Stochastic Gradient Descent

The most important feature of SGD is that it "computes the gradient and updates the weight matrix W on small batches of training data, rather than the entire training set itself." [SGD with Python](https://www.pyimagesearch.com/2016/10/17/stochastic-gradient-descent-sgd-with-python/)

Key vocabulary: 
- Stochastic Gradient Descent
- simulated annealing
- learning schedule
- epoch

### Mini-batch Gradient Descent

Key vocabulary: 
- mini-batch gradient descent
- mini-batches

## Polynomial Regression

Key vocabulary: 
- polynomial regression

## Learning Curve

Key vocabulary: 
- learning curves
- the bias/variance trade-off

## Regularized linear Models

### Ridge Regression

Key vocabulary: 
- ridge regression
- regularization term
- ridge regression cost function
- identity matrix
- ridge regression closed-form solution

### Lasso Regression

Key vocabulary: 
- least absolute shrinkage and selection operator regression <- just call is Lasso Regression
- lasso regression cost function
- sparse model
- subgradient vector

### Elastic Net

Key vocabulary: 
- elastic net cost function

### Early Stopping

Key vocabulary: 
- early stopping

## Logistic Regression

Key vocabulary: 
- logistic regression & logit regression (same thing)
- positive class
- negative class

### Estimating Probabilities

Key vocabulary: 
- logistic
- logistic regression model estimtated probability (vectorized form)
- sigmoid function
- logistic Regression model prediction

### Training and Cost Function

Key vocabulary: 
- cost function of a single training instance
- log loss
- logistic regression cost function (log loss)
- logistic cost function partial derivatives

### Decision Boundaries

Key vocabulary: 
- decision boundary

### Softmax Regression

Key vocabulary: 
- softmax regression & multinomial logistic regression <- same thing
- softmax function & normalized exponential <- same thing
- softmax score for class k
- softmax regression classifier prediction
- cross entropy
- cross entropy function
- corss entropy gradient vector for class k

___

#### TABLED ITEMS

- stochastic gradient descent, how random is it? 
  - pending
- classification vs regression models/algorithms/logic debate
  - [Difference between classifiaction and regression](https://techdifferences.com/difference-between-classification-and-regression.html)
  - [Classification vs Regression](https://www.geeksforgeeks.org/ml-classification-vs-regression/) (refer to chart at the bottom)
  - This [article](https://www.geeksforgeeks.org/understanding-logistic-regression/) explains what a logistic regression is. Just like a linear regression, logsitic regression is a linear model. "Logistic regression becomes a classification technique only when a decision threshold is brought into the picture. The setting of the threshold value is a very important aspect of Logistic regression and is dependent on the classification problem itself."
- learning curve, what's the purpose?
  - pending
- Andrew Ng's logistic regression
  - [link 1](https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning) Luis
  - [link 2](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN) Junwei's 
- Softmax, what is it and how are we going to use it? 
  - pending

#### GROUP WORK
- Pending

#### STATISTICS
- Pending

#### PYTHON
- Pending

#### MATHS
- Pending

#### LINKS
- Pending

#### BLOGS
- Pending

#### DATASETS
- Pending
