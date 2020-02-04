# Chapter 6 Decision Trees

- [Training and Visualizing a Decision Tree](#training-and-visualizing-a-decision-tree)
- [Making Predictions](#making-predictions)
- [Estimating Class Probabilities](#estimating-class-probabilities)
- [The CART Training Algorithm](#the-cart-training-algorithm)
- [Computational Complexity](#computational-complexity)
- [Gini Impurity or Entropy](#gini-impurity-or-entropy)
- [Regularization Hyperparameters](#regularization-hyperparameters)
- [Regression](#regression)
- [Instability](#instability)

## Training and Visualizing a Decision Tree

Decision Trees are versatile ML algorithms that can perform both classification and regression tasks, and even multioutput taks. They're also fundamental components of Random Forests - another ML algorithm. 

Key Vocabulary:
- [decision trees](https://en.wikipedia.org/wiki/Decision_tree): a decision support tool that uses a tree-like model of decisions and possible consequences, including chance event outcomes, resourece costs, and utility. 
- [random forest](https://en.wikipedia.org/wiki/Random_forest): are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training and outputting the class that is the mode of the classess (classification) or mrean prediction (regression) of the individual trees.

## Making Predictions

The root node is at the top and as you go down they're called a left/right child node. A leaf node does not have any child nodes. "One of the many qualities of Decision Trees is that they require very little data preparation... they don't require feature scaling or centering at all." A node's "gini" attribute measures its impourity: a node is pure if gini = 0 & if all training instances it applies to belong to the same class. 

Check out Gini impurity equation. 

Scikit-Learn uses CART algorithm - producing only binary trees (produces only 2 children). There are other tools/packages with more than one children.

White box vs black box: white box: easy to understand (decision trees), black box: difficult to interpret (Random Forests or neural network). 

Key Vocabulary: 
- [root node](https://en.wikipedia.org/wiki/Tree_(data_structure)) 
- [leaf node](https://en.wikipedia.org/wiki/Tree_(data_structure)#Terminology)
- [impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)

## Estimating Class Probabilities

Just know that a decision tree can estimate the probability that an instance belongs to a particular class k. 

## The CART Training Algorithm

What you need to know is that the algorithm splits the training set into two subjsets using 1) a single feature & 2) a threshold. How to select a feature and threshold? You'll want to produce the purest subsets. It does this recursively and only stops when it hits the maximum depth. 

Key Vocabulary:
- [Classification and Regression Tree (CART)](https://en.wikipedia.org/wiki/Predictive_analytics#Classification_and_regression_trees_(CART))
- [greedy algorithm](https://en.wikipedia.org/wiki/Greedy_algorithm)

## Computational Complexity

Just know that Decision Trees have an overall complexity of O(log2(m)). You can slow it down by presorting the data in SciKit-Learn.

## Gini Impurity or Entropy?

"In ML, entropy is frequently used as an impurity measure: a set's entropy is zero when it contains instances of only one class." There is an equation. 

So which one should you use? It really doesn't make a big differnece - they both lead to similiar trees. Gini is usually faster. Gini tends to isolate the most frequent class in its own branch of the tree, while entropy tends to produce slighty more balanced trees. 

## Regularization Hyperparameters

To avoid overfitting the training data, you need to restric the Decision Tree's freedom during training - called regularization. In Scikit-Learn, this is controlled by the "max_depth" hyperparameter. "min_samples_split", "min_samples_leaf", "min_weight_fraction_leaf", and "max_leaf_nodes" are all parameters found in the "DecisionTreeClassifier."

Key Vocabulary:
- [nonparametric model](https://en.wikipedia.org/wiki/Nonparametric_statistics): a unconstrained tree structure that adapts itself to the training model, most likely overfitting it. 
- [parametric model](https://en.wikipedia.org/wiki/Parametric_model): more like a linear model, has predetermined number of parameters, so its degree of freedom is limited, reducing the risk of overfitting. 

## Regression

Use the "DecisionTreeRegressor" to perform regression tasks in Decision Trees. Instead of a class in each node, it predicts a value. Just like a classification task, Decision trees are prone to overfitting when dealing with regressor tasks. To control this, you may want to try "min_samples_leaf" until the model looks more reasonable. 

## Instability



___

#### TABLED ITEMS
- Pending

#### GROUP WORK
- Pending

#### STATISTICS
- Pending

#### PYTHON
- Pending

#### MATHS
- Pending

#### LINKS
- I like [link](https://medium.com/@ankitnitjsr13/math-behind-support-vector-machine-svm-5e7376d0ee4d) from Medium that attempts to explain the math in simple terms. It covers the basic from a classification standpoint.

#### BLOGS
- Pending

#### DATASETS
- Pending
