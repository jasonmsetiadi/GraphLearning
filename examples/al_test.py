import graphlearning.active_learning as al
import graphlearning as gl
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

X,labels = datasets.make_moons(n_samples=500,noise=0.1)
W = gl.weightmatrix.knn(X,10)
train_ind = gl.trainsets.generate(labels, rate=5)
#plt.scatter(X[:,0],X[:,1], c=labels)
#plt.scatter(X[train_ind,0],X[train_ind,1], c='r')
#plt.show()

model = gl.ssl.laplace(W)
act = al.active_learning(W, train_ind, labels[train_ind], eval_cutoff=200)
methods = [al.uncertainty_sampling(), al.v_opt(), al.sigma_opt(), al.model_change(), al.model_change_vopt()]
names = ["UNCERTAINTY","VOPT","SIGMA OPT","MODEL CHANGE","MC VOPT"]
for i in range(len(methods)):
    acq = methods[i]
    print(names[i])

    for i in range(10):
        u = model.fit(act.current_labeled_set, act.current_labels) # perform classification with GSSL classifier
        query_points = act.select_query_points(acq, u, oracle=None) # return this iteration's newly chosen points
        query_labels = labels[query_points] # simulate the human in the loop process
        act.update_labeled_data(query_points, query_labels) # update the active_learning object's labeled set

        # plot
        #plt.scatter(X[:,0],X[:,1], c=labels)
        #plt.scatter(X[act.current_labeled_set,0],X[act.current_labeled_set,1], c='r')
        #plt.scatter(X[query_points,0],X[query_points,1], c='r', marker='*', s=200, edgecolors='k', linewidths=1.5)
        #plt.show()
        print(act.current_labeled_set)
        print(act.current_labels)

    # reset active learning object    
    print("reset")
    act.reset_labeled_data()
    print(act.current_labeled_set)
    print(act.current_labels)


