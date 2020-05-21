import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
start_time = time.time()

def subgradient_descent(x,y,initial_values,threshold=0.001):
    w_initial,b=initial_values
    iteration=0
    C=1
    difference=w_initial
    ncol=x.shape[1]
    while((np.linalg.norm(difference)>threshold)&(iteration<10000)):
        '''Learning rate from Frank Wolfe Algorithm'''
        gamma=2/(iteration+1)
        z=np.stack([y]*ncol,axis=1)
        idx_subgrad=np.where(np.multiply(np.add(np.matmul(x, w_initial), b),y)<1.0, -1, 0)
        subgrad_w = np.dot(idx_subgrad, np.multiply(z, x))
        #subgrad_w=np.dot(idx_subgrad,np.multiply(np.stack((y,y),axis=1),x))
        subgrad_b=np.sum(np.multiply(idx_subgrad,y))

        w_update=w_initial-gamma*(w_initial+C*subgrad_w)
        b_update=b-gamma*C*subgrad_b
        difference=w_update-w_initial
        w_initial=w_update
        b=b_update
        iteration+=1

    return (w_update,b_update)

if __name__ == "__main__":
    breast_cancer = load_breast_cancer()
    x = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names).to_numpy(copy=True) #Input x-data
    y = pd.DataFrame(breast_cancer.target).to_numpy(copy=True).reshape((x.shape[0],)) #Input y-data

    w_initial= np.ones((x.shape[1],)) #initial guess for slope
    b_initial= 10  #initial guess for constant
    initial_values = (w_initial, b_initial)

    w_final,b_final=subgradient_descent(x,y,initial_values)
    print(w_final)
    print(b_final)

print("--- Subgradient descent for primal SVM problem took %s seconds ---" % round((time.time() - start_time),3))