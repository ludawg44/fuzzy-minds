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

Let me give this a shot: sometimes when you're bagging, you're only bagging a subset of the total dataset. For instance, you were bagging and pasting 67% of the total dataset. The remaining 37% of the training instances that weren't sampled are considered "out-of-bag" (oob) instances. I think you test the ensemble on this oob set and compare it to your bagging classifier accuracy. They should be close. 

## Random Patches and Random Subspaces

I believe this technique works best with high-dimensional inputs (such as images). 

Key vocabulary: 
- random patches methods: 

## Random Forests

Random Forest is an ensemble of Decision Trees. You can use the ```RandomForestClassifier``` class. 

"The Random Forest algorithm introduces extra randomness when growing trees; instead of searching for the very best feature when splitting a node, it searches for the best feature among a random subset of features."

As Random Forest algorithm increases in diversity -> so does the higher bias for a lower variance

## Extra Trees

"When you are growing a tree in a Random Forest, at each node only a random subset of the features is considered for splitting. It is possible to amke trees even more random by also using random thresholds fore ach feature rather thean searching for the best possible thresholds (like Decision Trees do)."

Extra trees is short for Extremely Randomized Trees ensblem. Again, you trade more bias for a lower variance. 

It is hard to tell whether the ```RandomForestClassifier``` is fast than the ```ExtraTreesClassifier```. You have to try both. 

## Feature Importance

"Random Forests are very handy to get a quick understanding of what features actually matter, in particular if you need to perform feature selection."

## Boosting

Key Vocabulary:
- [boosting or hypothesis boosting](https://en.wikipedia.org/wiki/Boosting_(machine_learning)): refers to any Ensemble method that can combine several weak learners into a strong learner. 

## AdaBoost

## Gradient Boosting

## Stacking 

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
- https://towardsdatascience.com/overview-ensemble-learning-made-simple-d4ac0d13cb96

#### BLOGS
- Pending

#### DATASETS
- Pending
