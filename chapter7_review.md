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

"Ensemble methods work best when the predictors are as independent from one another as possible. on eway to get diverse classifiers is to train them using very different algorithms. This increases the chance that they make different types of errors, improving the ensemble's accuracy."

Key vocabulary:
- hard voting: a majority classifer
- weaker learner: it does only slightly better than random guessing
- strong learner: achieving high accuracy
- law of large numbers: 
- soft voting:

## Bagging and Pasting

Important distinction: both bagging and pasting allow training instances to be sampled several times across multiple predictors, but only bagging allows training instances to be sampled several times for the same predictor. 

Predictors can all be trained in parallel - via different CPU cores and different servers. Predictions can also be made in parallel. This is one of the reasons bagging and pasting are such popular methods: they scale very well. 

Key vocabulary:
- statistical model: the most frequent prediction, just like a hard voting classifier. 

## Bagging and Pasting in Scikit-Learn

Look at the book, there's a lot of code.

```
from sklearn.ensblem import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
  DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True, n_jobs=-1)
  bag_clf.fit(X_train, y_train)
  y_pred = bag_clf.predict(X_test)
  
```

## Out-of-bag Evaluation

## Random Patches and Random Subspaces

## Random Forests

## Extra Trees

## Feature Importance

## Boosting

## AdaBoost

## Gradient Boosting

## Stacking 
