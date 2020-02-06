# Chapter 7 Ensemble Learning and Random Forests

- [Voting Classifiers](#voting-classifiers)
- [Bagging and Pasting](#bagging-and-pasting)
- [Bagging and Pasting in Scikit-Learn](#bagging-and-pasting-in-scikit-learn)
- [Out-of-Bag Evaluation](#out-of-bag-evaluation)
- [Random Patches and Random Subspaces](#random-patches-and-random-subspaces)
- [Random Forests](#random-forests)
- [Extra-Trees](#extra-trees)
- [Feature Importance](#feature-importance)
- [Boosting](#boosting)
- [AdaBoost](#adaboost)
- [Gradient Boosting](#gradient-boosting)
- [Stacking](#stacking)

Key vocabulary:
- ensemble: A group of dictors
- [Ensemble Learning](https://en.wikipedia.org/wiki/Ensemble_learning): In statistics and machine learning, ensemble methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone. 
- [Random Forest](https://en.wikipedia.org/wiki/Random_forest): An ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees
- bagging: Involves having each model in the ensemble vote with equal weight.
- boosting: Involves incrementally building an ensemble by training each new model instance to emphasize the training instances that previous models mis-classified.
- stacking: Involves training a learning algorithm to combine the predictions of several other learning algorithms.

## Voting Classifiers

Key vocabulary:
- hard voting:
- weaker learner: 
- strong learner: 
- law of large numbers: 
- soft voting:

## Bagging and Pasting

Important distinction: both bagging and pasting allow training instances to be sampled several times across multiple predictors, but only bagging allows training instances to be sampled several times for the same predictor. 

Predictors can all be trained in parallel - via different CPU cores and different servers. Predictions can also be made in parallel. This is one of the reasons bagging and pasting are such popular methods: they scale very well. 

Key vocabulary:
- statistical model: the most frequent prediction, just like a hard voting classifier. 

## Bagging and Pasting in Scikit-Learn

## Out-of-bag Evaluation

## Random Patches and Random Subspaces

## Random Forests

## Extra Trees

## Feature Importance

## Boosting

## AdaBoost

## Gradient Boosting

## Stacking 
