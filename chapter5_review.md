# Chapter 5 Support Vector Machines

- [Linear SVM Classification](#linear-svm-classification)
  - [Soft Margin Classification](#soft-margin-classification)
- [Nonlinear SVM Classification](#nonlinear-svm-classification)
  - [Polynomial Kernel](#polynomial-kernel)
  - [Similarity Features](#similarity-features)
  - [Gaussian RBF Kernel](#gaussian-rbf-kernel)
  - [Computational Complexity](#computational-complexity)
- [SVM Regression](#svm-regression)
- [Under the Hood](#under-the-hood)
  - [Decision Function and Predictions](#decision-function-and-predictions)
  - [Training Objective](#training-objective)
  - [Quadratic Programming](#quadratic-programming)
  - [The Dual Problem](#the-dual-problem)
  - [Kernelized SVMs](#kernelized-svms)
  - [Online SVMs](#online-svms)

## Linear SVM Classification

SVMs are sensitive to feature scales. There's an image that shows a really good example of unscaled and scaled features. In order to apply this in code, use Scikit-Learn's StandardScaler. 

Key vocabulary: 
- [Support Vector Machine](https://en.wikipedia.org/wiki/Support-vector_machine) (SVM): SVMs are powerful and versatile. They are capable of both classification and regression modeling. 
- [linearly separable](https://en.wikipedia.org/wiki/Linear_separability#): when two classes can clearly be separated easily with a straight line. 
- large margin classification: fits the widest possible lines across two classes. 
- support vectors: fully determined by the instances located at the edge of the "street"

### Soft Margin Classification

If your SVM model is overfitting, you can try to regularize it by lowering C. SVM classifiers do not output probabilities for each class. Don't forget to use the "loss = 'hing'" hyperparameter as this is not the default value. For better performance, set "dual=False" hyperparameter to False, unless there are more features than training instances. 

Key vocabulary:
- hard margin classification
- margin violations
- soft margin classification
- hinge loss

## Nonlinear SVM Classification

Python: SciKit-Learn's "Pipeline" & "PolynomialFeatures" from "sklearn.pipeline" & "sklearn.preprocessing" from chapter 4. 

### Polynomial Kernel

Key vocabulary:
- kernel trick: makes it possoible to get the same result as if you had added many polynomial features, even with very high-degree polynomials, without actually having to add them. 

### Similarity Features

Key vocabulary:
- similarity function
- landmark
- Radial Basis Function (RBF)

### Gaussian RBF Kernel

Key vocabulary:
- string kernels
- Levenshtein distance

### Computational Complexity

Key vocabulary:
- sparse features: when each instance has few nonzero features

## SVM Regression

## Under the Hood

### Decision Function and Predictions

Key vocabulary:
- Linear SVM classifier prediction 

### Training Objective

Key vocabulary:
- Hard margin linear SVM classifier objective
- slack variable
- soft margin linear SVM classifier objective

### Quadratic Programming

Key vocabulary: 
- Quadratic Programming problem

### The Dual Problem

Key vocabulary:
- primal problem
- dual problem
- Dual form of the linear SVM objective
- From the dual solution to the primal solution

### Kernelized SVMs

Key vocabulary:
- second-degree polynomial mapping
- Kernel trick for a second-degree polynomial mapping
- Kernel
- Common Kernels
  - Linear
  - Polynomial
  - Gaussian RBF
  - Sigmoid
- Mercer's Theorem
- Making predictions with a kernelized SVM
- Using the kernel trick to compute the bias term

### Online SVMs

Key vocabulary: 
- Linear SVM classifier cost function
- Hing Loss
- subderivative

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
- Pending

#### BLOGS
- Pending

#### DATASETS
- Pending

