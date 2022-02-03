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

#Directories
results_dir = os.path.join(os.getcwd(),'results')

class active_learning:
    def __init__(self, W, training_set, current_labeled_set, spectral_truncation_parameter, gamma=0.1):
        self.graph = W
        self.current_labeled_set = current_labeled_set
        self.training_set = training_set
        self.candidate_inds = np.setdiff1d(training_set, current_labeled_set)
        self.spectral_truncation_parameter = spectral_truncation_parameter
        self.gamma = gamma
        evals, self.evecs = gl.graph(W).eigen_decomp(normalization='normalized', k=spectral_truncation_parameter)
        self.covariance_matrix = np.linalg.inv(np.diag(evals) + self.evecs[current_labeled_set,:].T @ self.evecs[current_labeled_set,:] / gamma**2.)
  
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

        """

        for i in range(batch_size):
            objective_values = acquisition.compute_acquisition_function_values(ssl, self, labels, uncertainty_method='smallest_margin')
            new_train_ind = self.candidate_inds[np.argmax(objective_values)]
            self.current_labeled_set = np.append(self.current_labeled_set, new_train_ind)
            self.candidate_inds = np.setdiff1d(self.training_set, self.current_labeled_set)
            for k in [new_train_ind]:
                vk = self.evecs[k]
                Cavk = self.covariance_matrix @ vk
                ip = np.inner(vk, Cavk)
                self.covariance_matrix -= np.outer(Cavk, Cavk)/(self.gamma**2. + ip)


class acquisition_function:
    @abstractmethod
    def compute_acquisition_function_values(self, ssl, active_learning, labels, uncertainty_method):
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
        uncertainty_method : str
            if method requires "uncertainty calculation" this string specifies which type of uncertainty measure to apply

        Returns
        -------
        acquisition_values : numpy array, float
            acquisition function values
        """
        raise NotImplementedError("Must override compute_acquisition_function_values")

class uncertainty_sampling(acquisition_function):
    """Uncertainty Sampling
    ===================

    Active learning algorithm that selects points that the classifier is most uncertain of.

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
    acq = uncertainty_sampling()
    al = active_learning(W, np.arange(len(labels)), train_ind, 300)
    al.select_next_training_points(model, acq, labels, batch_size=5)
    new_ind = np.setdiff1d(al.current_labeled_set, train_ind)

    plt.scatter(X[:,0],X[:,1], c=labels)
    plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
    plt.scatter(X[new_ind,0],X[new_ind,1], c='r', marker='*', s=200, edgecolors='k', linewidths=1.5)
    plt.show()
    ```

    Reference
    ---------
    """
    def compute_acquisition_function_values(self, ssl, active_learning, labels, uncertainty_method='smallest_margin'):
        u = ssl.fit(active_learning.current_labeled_set, labels[active_learning.current_labeled_set])
        if uncertainty_method == "norm":
            u_probs = softmax(u[active_learning.candidate_inds], axis=1)
            one_hot_predicted_labels = np.eye(u.shape[1])[np.argmax(u[active_learning.candidate_inds], axis=1)]
            unc_terms = np.linalg.norm((u_probs - one_hot_predicted_labels), axis=1)
        elif uncertainty_method == "entropy":
            u_probs = softmax(u[active_learning.candidate_inds], axis=1)
            unc_terms = np.max(u_probs, axis=1) - np.sum(u_probs*np.log(u_probs +.00001), axis=1)
        elif uncertainty_method == "least_confidence":
            unc_terms = np.ones((u[active_learning.candidate_inds].shape[0],)) - np.max(u[active_learning.candidate_inds], axis=1)
        elif uncertainty_method == "smallest_margin":
            u_sort = np.sort(u[active_learning.candidate_inds])
            unc_terms = 1.-(u_sort[:,-1] - u_sort[:,-2])
        elif uncertainty_method == "largest_margin":
            u_sort = np.sort(u[active_learning.candidate_inds])
            unc_terms = 1.-(u_sort[:,-1] - u_sort[:,0])
        return unc_terms

class v_opt(acquisition_function):
    """Variance Optimization
    ===================

    Active learning algorithm that selects points that minimizes the variance of the distribution of unlabeled nodes

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
    acq = v_opt()
    al = active_learning(W, np.arange(len(labels)), train_ind, 300)
    al.select_next_training_points(model, acq, labels, batch_size=5)
    new_ind = np.setdiff1d(al.current_labeled_set, train_ind)

    plt.scatter(X[:,0],X[:,1], c=labels)
    plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
    plt.scatter(X[new_ind,0],X[new_ind,1], c='r', marker='*', s=200, edgecolors='k', linewidths=1.5)
    plt.show()
    ```

    Reference
    ---------
    """
    def compute_acquisition_function_values(self, ssl, active_learning, labels, uncertainty_method='smallest_margin'):
        Cavk = active_learning.covariance_matrix @ active_learning.evecs[active_learning.candidate_inds,:].T
        col_norms = np.linalg.norm(Cavk, axis=0)
        diag_terms = (active_learning.gamma**2. + np.array([np.inner(active_learning.evecs[k,:], Cavk[:, i]) for i,k in enumerate(active_learning.candidate_inds)]))
        return col_norms**2. / diag_terms

class sigma_opt(acquisition_function):
    """Sigma Optimization
    ===================

    Active learning algorithm that selects points that minimizes the sum of the associated entries in the covariance matrix

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
    acq = sigma_opt()
    al = active_learning(W, np.arange(len(labels)), train_ind, 300)
    al.select_next_training_points(model, acq, labels, batch_size=5)
    new_ind = np.setdiff1d(al.current_labeled_set, train_ind)

    plt.scatter(X[:,0],X[:,1], c=labels)
    plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
    plt.scatter(X[new_ind,0],X[new_ind,1], c='r', marker='*', s=200, edgecolors='k', linewidths=1.5)
    plt.show()
    ```

    Reference
    ---------
    """
    def compute_acquisition_function_values(self, ssl, active_learning, labels, uncertainty_method='smallest_margin'):
        Cavk = active_learning.covariance_matrix @ active_learning.evecs[active_learning.candidate_inds,:].T
        col_sums = np.sum(Cavk, axis=0)
        diag_terms = (active_learning.gamma**2. + np.array([np.inner(active_learning.evecs[k,:], Cavk[:, i]) for i,k in enumerate(active_learning.candidate_inds)]))
        return col_sums**2. / diag_terms

class model_change(acquisition_function):
    """Model Change
    ===================

    Active learning algorithm that selects points that will produce the greatest change in the model

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
    acq = model_change()
    al = active_learning(W, np.arange(len(labels)), train_ind, 300)
    al.select_next_training_points(model, acq, labels, batch_size=5)
    new_ind = np.setdiff1d(al.current_labeled_set, train_ind)

    plt.scatter(X[:,0],X[:,1], c=labels)
    plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
    plt.scatter(X[new_ind,0],X[new_ind,1], c='r', marker='*', s=200, edgecolors='k', linewidths=1.5)
    plt.show()
    ```

    Reference
    ---------
    """
    def compute_acquisition_function_values(self, ssl, active_learning, labels, uncertainty_method='smallest_margin'):
        unc_terms = uncertainty_sampling().compute_acquisition_function_values(ssl, active_learning, labels)
        Cavk = active_learning.covariance_matrix @ active_learning.evecs[active_learning.candidate_inds,:].T
        col_norms = np.linalg.norm(Cavk, axis=0)
        diag_terms = (active_learning.gamma**2. + np.array([np.inner(active_learning.evecs[k,:], Cavk[:, i]) for i,k in enumerate(active_learning.candidate_inds)]))
        return unc_terms * col_norms / diag_terms  

class model_change_vopt(acquisition_function):
    """Model Change Variance Optimization
    ===================

    Active learning algorithm that is a combination of Model Change and Variance Optimization

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
    acq = model_change_vopt()
    al = active_learning(W, np.arange(len(labels)), train_ind, 300)
    al.select_next_training_points(model, acq, labels, batch_size=5)
    new_ind = np.setdiff1d(al.current_labeled_set, train_ind)

    plt.scatter(X[:,0],X[:,1], c=labels)
    plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
    plt.scatter(X[new_ind,0],X[new_ind,1], c='r', marker='*', s=200, edgecolors='k', linewidths=1.5)
    plt.show()
    ```

    Reference
    ---------
    """
    def compute_acquisition_function_values(self, ssl, active_learning, labels, uncertainty_method='smallest_margin'):
        unc_terms = uncertainty_sampling().compute_acquisition_function_values(ssl, active_learning, labels)
        Cavk = active_learning.covariance_matrix @ active_learning.evecs[active_learning.candidate_inds,:].T
        col_norms = np.linalg.norm(Cavk, axis=0)
        diag_terms = (active_learning.gamma**2. + np.array([np.inner(active_learning.evecs[k,:], Cavk[:, i]) for i,k in enumerate(active_learning.candidate_inds)]))
        return unc_terms * col_norms **2. / diag_terms
