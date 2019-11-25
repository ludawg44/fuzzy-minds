# Hands On Machine Learning Chapter Overview
A quick review of the subject and topics covered in each capter with links to help solidify the material. 

## Chapter 3
**SUMMARY**: This chapter focuses on **classification systems** and how to **evaluate them**. One common way to evaluate a classification system is to use a confusion matrix. Another method is the precsion score, the recall score, understanding the precision/recall tradeoff, and the ROC Curve. He does this for a binary classifier and then multiclass classification.

1. MNIST 
- Use the built in dataset and choose one digit for the binary classification example
- Display the image
- Create a train and test set before inspecting any data. Put the test data aside until the very end.

2. Training a Binary Classifier
- The first model he covers in this chapter is the Stochastic Gradient Descent. For a review, check the Maths section below.
- Learn this by heart:
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
"A good way to evaluate a model is to use cross-validation." There is a long way, which definitely serve's its purpose or a short way.
Let's explore the short way:
```
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
```
  **Confusion Matrix**

  **Precision and Reacll**

  **Precision/Recall Trade-off**

  **The ROC Curve**

4. Multiclass Classification

5. Error Analysis

6. Multilable Classification

7. Multioutput Classification

8. Exercises



**STATISTICS**: 
- Classification Systems: (definition)
- You want to think about classification systems in two ways: binary classification & multiclass classification
- Linear classifier algorithms: refer to wikipedia
- For unsupervised, supervised & semi-supervised learning algorithms: refer to wikipedia
- How does this relate to deep learning? 
  - Deep learning uses binary and muliclass classificaitons in their input and hidden layers

**PYTHON**: 
- fetch_openml in sklearn.dataset
- know how to separate train and test sets. Usually in this format:
  ```
  X_train, y_train, X_test, y_test = ...
  ```
- 


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


