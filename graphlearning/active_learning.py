"""
Active Learning
========================

This module implements many active learning algorithms in an objected-oriented
fashion, similar to [sklearn](https://scikit-learn.org/stable/). The usage is similar for all algorithms.
Below, we give some high-level examples of how to use this module. There are also examples for some
individual functions, given in the documentation below.

"""
import numpy as np
import graphlearning as gl
from scipy.special import softmax
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt

from . import utils 
from . import graph

#Directories
results_dir = os.path.join(os.getcwd(),'results')

class active_learning:
    def __init__(self, W, training_set, current_labeled_set):
        self.graph = W
        self.current_labeled_set = current_labeled_set
        self.training_set = training_set
        self.candidate_inds = np.setdiff1d(self.training_set, self.current_labeled_set)

    def select_next_training_points(self, ssl, acquisition, labels, batch_size=1):
        """Select next training points
        ======
    
        Select "batch_size" number of points to be labeled by an active learning algorithm we specify 

        Parameters
        ----------
        ssl : class object 
            ssl classifier object. 
        acquisition : class object
            acquisition function object.
        labels : numpy array, int
            True labels for all datapoints.
        batch_size : int (optional), default=1
            number of points want to be labeled in one iteration of active learning.

        Returns
        -------
        new labeled set : numpy array (int)
            new labeled set after active learning.
        """

        for i in range(batch_size):
            objective_values = acquisition.compute_acquisition_function_values(ssl, self, labels)
            new_train_ind = self.candidate_inds[np.argmax(objective_values)]
            self.current_labeled_set = np.append(self.current_labeled_set, new_train_ind)
            self.candidate_inds = np.setdiff1d(self.training_set, self.current_labeled_set)

        return self.current_labeled_set

class acquisition_function:
    @abstractmethod
    def compute_acquisition_function_values(self, ssl, active_learning, labels):
        """Internal Compute Acquisition Function Values Function
        ======

        Internal function that any acquisition function object must override. 

        Parameters
        ----------
        ssl : class object 
            ssl classifier object. 
        active_learning : class object
            active learning object.
        labels : numpy array, int
            True labels for all datapoints.

        Returns
        -------
        acquisition_values : numpy array, float
            acquisition function values
        """
        raise NotImplementedError("Must override compute_acquisition_function_values")

class uncertainty_sampling(acquisition_function):
    """Uncertainty Sampling
    ===================

    Active learning algorithm that selects points to label based on what the classifier is most uncertain of.

    Parameters
    ----------
    ssl : class object 
        ssl classifier object. 
    active_learning : class object
        active learning object.
    labels : numpy array, int
        True labels for all datapoints.

    Returns
    -------
    uncertainty_terms : numpy array, float
        acquisition function values of uncertainty sampling

    Examples
    --------
    ```py
    import numpy as np
    import graphlearning as gl
    import matplotlib.pyplot as plt
    import sklearn.datasets as datasets

    X,labels = datasets.make_moons(n_samples=500,noise=0.1)
    W = gl.weightmatrix.knn(X,10)
    train_ind = gl.trainsets.generate(labels, rate=5)

    plt.scatter(X[:,0],X[:,1], c=labels)
    plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
    plt.show()

    model = gl.ssl.laplace(W)
    uncertainty = uncertainty_sampling()
    al = active_learning(W, np.arange(len(labels)), train_ind)
    al.select_next_training_points(model, uncertainty, labels, batch_size=1)
    new_ind = np.setdiff1d(al.current_labeled_set, train_ind)

    plt.scatter(X[:,0],X[:,1], c=labels)
    plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
    plt.scatter(X[new_ind,0],X[new_ind,1], c='r', marker='*', s=200, edgecolors='k', linewidths=1.5)
    plt.show()
    ```

    Reference
    ---------
    """

    def compute_acquisition_function_values(self, ssl, active_learning, labels):
        u = ssl.fit(active_learning.current_labeled_set, labels[active_learning.current_labeled_set])
        u_probs = softmax(u[active_learning.candidate_inds], axis=1)
        one_hot_predicted_labels = np.eye(u.shape[1])[np.argmax(u[active_learning.candidate_inds], axis=1)]
        uncertainty_terms = np.linalg.norm((u_probs - one_hot_predicted_labels), axis=1)

        return uncertainty_terms

