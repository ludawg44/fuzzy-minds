#Chapter 3

**SUMMARY**: This chapter focuses on **classification systems** and how to **evaluate them**. One common way to evaluate a classification system is to use a confusion matrix. Another method is the precsion score, the recall score, understanding the precision/recall tradeoff, and the ROC Curve. The author does this for binary classifiers and then multiclass classifications.

1. MNIST 
- Use the built in dataset and choose one digit for the binary classification example
- Display the image
- Create a "train" and "test" set before inspecting any data. Put the "test" data aside until the very end.

2. Training a Binary Classifier
- The first model he covers in this chapter is the Stochastic Gradient Descent. For a review, check the Maths section below.
- We put one image in the y_train_5 variable
- Become familiar with this structure:
```
from sklearn.linear_model import SGDClassifier

# TRAIN
# 'random_state' let's you reproduce your results. y_train_5 is using a binary classifer
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# PREDICT
# This takes the first 
sgd_clf.predict([take a digit from the set and create a variable to test])
```
3. Performance Measures
  **Measuring Accuracy Using Cross-Validation**
"A good way to evaluate a model is to use cross-validation." There is a long way, which definitely serves its purpose, and a short way.
Let's explore the short way:
```
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
```
  **Confusion Matrix**
- In the confusion matrix, the axises are labeled "Actual" and "Predicted"(refer to Wikipedia down below for reference). We have our "actual" digit from "y_train_5." In order to get our "predicted" scores, you have run this code:
```
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
```
- Alright, we have both y_train_5 and y_train_pred. Let's run the confusion matrix
```
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)
```
  **Precision and Recall**
- A precision score of 0.75 means your model was only 75% accurate
- A recall score of 0.68 means your model only detects 68% of the digits (images, clothes, faces, whatever you're classifing) that you selected
```
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)
```
- He briefingly mentions F1 scores. I'm not going to cover it here. 

  **Precision/Recall Trade-off**
- Increasing precision reduces recall & increasing recall reduces precision. You can't have it both ways.
- To understand the tradeoff, you have to realize how the SGDClassifier() function makes a classification decision. In scikit-learn, each instance gets a score based on the decision function. That score, from the decision function, is compared to a threshold number. If it is above the threshold, it'll be a positive class. If its below, it gets put in a negative class. SGDClassifier in scikit-learn has a default threshold of 0.
- Sci-kit Learn doesn't allow you to change the decision function, but it allows you to change the threshold. There is a lot going on here, so review on your own. 
```
threshold = 8000
y_some_digit_pred = (y_scores > threshold)
# Print it out
y_some_digit_pred
```
- Why use 8,000 in the book? You can figure out by using another function that we used, just change the parameter
```
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, csv=3, method='decision_function")
```
- Now you can use the precision_recall_curve() function to find all possible thresholds and graph it. But I won't go over that here, refer to book's Github account

  **The ROC Curve**
- This is an exciting chapter because not only to get to graph more performance measurements covered in HBAP, but we're going to compare the SGDClassifier with another one!
- To plot the ROC curve, you need the TPR and FPR values for the various thresholds. You have the threshold from up above. 
```
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
```
- I won't be going over plotting the graphs here. But let's get the ROC AUC score now:
```
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)
```
- WOOT WOOT! We're going to compare two classifers now! I'm just going to provide the basic code to get a score. Refer to the author's GitHub account if you want to see how to graph it. 
```
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

# "The roc_curve() function expects labels and scores, but instead of scores you can give it class probabilities.
y_scores_forest = y_probas_forest[:,1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

# COMPARE ROC AUC SCORES
roc_auc_score(y_train_5, y_scores_forest)
```

4. Multiclass Classification

5. Error Analysis

6. Multilabel Classification

7. Multioutput Classification

8. Exercises



**STATISTICS**: 
- Classification Systems: (definition)
- You want to think about classification systems in two ways: binary classification & multiclass classification
- Linear classifier algorithms: refer to wikipedia
- For unsupervised, supervised & semi-supervised learning algorithms: refer to wikipedia
- How does this relate to deep learning? 
  - Deep learning uses binary and multiclass classificaitons in their input and hidden layers

**PYTHON**: 
SciKit-Learn
- fetch_openml
- know how to separate train and test sets. Usually in this format:
  ```
  X_train, y_train, X_test, y_test = ...
  ```
Basic intuition: choose classifier model, get the function from scikit-learn, fit it, predict it
- SGDClassifier
- RandomForestClassifier

Measuring functions:
- cross_val_score
- cross_val_predict
- confusion_matrix
- precision_score
- recall_score


**MATHS**:
- Understaing the mathematics behinds [Gradient Descent](https://towardsdatascience.com/understanding-the-mathematics-behind-gradient-descent-dde5dc9be06e)


Links
- Wikipedia's page on [classification systems](https://en.wikipedia.org/wiki/Statistical_classification)
- Wikipedia's page on [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)

Blogs
- I found this blog by [Jack Mcknew](https://jmckew.com/2019/10/18/hands-on-machine-learning-chapter-3/) on chapter 3. It's a quick read. 

Data
- The MNIST dataset has been the introductory dataset for a classification for some time now. As the models have gotten more
sophisticaed, the optimization levels have been identical in both Deep Learning and using more simple Machine Learning models.
Thus the creation of [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist). 
- Someone work with me to classify this [dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

