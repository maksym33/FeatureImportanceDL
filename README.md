# FeatureImportanceDL
Contains example of a deep-learning based dual-net feature selection method from https://arxiv.org/abs/2010.08973 :
<img src="./graphics/model.png" width="300">

This is an embedded method for supervised tasks: after training, it is able to give predictions (better then vanilla architecture) while also return population-wise (global) feature importances.

See ```example.py``` script that shows how the method works for a basic XOR dataset.
Some examples for MNIST dataset (digit 3 & 8 differentiation):

![MNIST importance](./graphics/MNIST/mnist_full_importance.png)
![Top features](./graphics/MNIST/mnist_masked_importance.png)
![Superimposed on mean 3](./graphics/MNIST/mnist_3_importance.png)
![Superimposed on mean 8](./graphics/MNIST/mnist_8_importance.png)

