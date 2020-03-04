# Chapter 8 Dimensionality Reduction

- [The Curse of Dimensionality](#the-curse-of-dimensionality)
- [Main Approaches for Dimensionality Reduction](#main-approaches-for-dimensionality-reduction)
- [Projection](#projection)
- [Manifold Learning](#manifold-learning)
- [PCA](#PCA)
- [Preserving the Varience](#preserving-the-varience)
- [Principal Components](#principal-components)
- [Projecting Down to d Dimensions](#projecting-down-to-d-dimensional)
- [Using Scikit-Learn](#using-scikit-learn)
- [Explained Varience Ratio](#explained-varience-ratio)
- [Choosing the Right Number of Dimensions](#choosing-the-right-number-of-dimensions)

## The Curse of Dimensionality
- Point, segment, square, cube, and tesseract (0D to 4D hypercubes)
- The more dimensions the training set has, the greater the risk of overfitting it.
- [Wikipedia](https://en.wikipedia.org/wiki/Curse_of_dimensionality)

## Main Approaches for Dimensinoality Reduction
- 2 main approaches to reducgin dimensionality: projection & manifold learning

### Projection
- 

### Manifold Learning
- manifold
- manifold hypothesis

## PCA
- Principal Component Analysis (PCA) is by far the most popular dimensionality reduction algorithm. 
- [Principal Component Analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis)

### Preserving the Varience
- "Before you can project the training set onto a lower-dimensional hyperplane, you first need to choose the right hyperplane. 

## Principal Components
- [Single Value Decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition)
- Principal components matrix equation
- You're going to use NumPy's svd() function

# Projecting Down to d Dimension
1) Identify all the principal components
2) Reduce the dimensionality of the dataset down to d dimensions by projecting it onto the hyperplane defined by the first d principal components. 

# Using Scikit-Learn
- You're going to use Scikit-Learn's PCA class, that uses SVD decomposition to implement PCA. Here is an example: 

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
X2D = pca.fit_transform(x)

# Explained Variance Ratio
- [explained variance ratio](https://ro-che.info/articles/2017-12-11-pca-explained-variance)

# Choosing the Right Number of Dimensions

# PCA for Compression

# Randomized PCA

# Incremental PCA

# Kernel PCA

# Selecting a Kernel and Tuning Hyperparameters

# LLE

# Other Dimensionality Reduction Techniques


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
