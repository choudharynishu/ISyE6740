#Standard imports
import os
import math
import random
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer       #Popular classification dataset
from sklearn.model_selection import train_test_split  #For randomly dividing into training and testing data
address = '/Users/nishuchoudhary/Desktop/'

#This function is used to resample data from a given dataset using given probabilities
def resample_pr(train_X,train_y,weights):
    nsamples=train_X.shape[0]
    cumsum=np.cumsum(weights)
    train_X['cdf']=cumsum
    
    x_list=[]
    y_list=[]
    
    for row_index in range(nsamples):
        random_num = random.random()
        row = train_X[train_X['cdf'] > random_num].index.min()
        x_list.append(train_X.iloc[row, 0:30])
        y_list.append(train_y.iloc[row, :])
    
    resampled_x=pd.DataFrame(x_list)
    resampled_y=pd.DataFrame(y_list)
    
    del y_list
    del x_list

    resampled_x.reset_index(inplace=True,drop=True)
    resampled_y.reset_index(inplace=True,drop=True)

    return (resampled_x, resampled_y)

#A function to produce a weak classifier from a give feature
def weak_classifier(feature_x,train_y,weights):
    unique_values=feature_x.unique()
    numRows=feature_x.shape[0]
    y = train_y
    min_error=np.inf

    for test in unique_values:
        prediction = pd.DataFrame(index=range(numRows))
        prediction['values'] = 0
        prediction[feature_x[:] < test] = 1
        bool_list=(prediction.values!=y.values).reshape((numRows,))
        error=np.sum(weights[bool_list])
        if error > 0.5:
           error = 1 - error  
        if error < min_error:
           threshold=test
           min_error = error
        del prediction

    return (threshold,min_error)

def adaboost(train_X,train_y,nclf=3):
    weakclassifiers=[]
    thresholds=[]
    initial_wvalue = 1 / len(train_y)
    weights = initial_wvalue * np.ones((len(train_y),))
    for iteration in range(nclf):
        resampled_x,resampled_y=resample_pr(train_X,train_y,weights)
        nsamples,nfeatures=resampled_x.shape
        min_error=np.inf

        for feature_i in range(nfeatures):
            threshold,error=weak_classifier(resampled_x.iloc[:,feature_i],resampled_y,weights)
            if error<min_error:
               min_error=error
               selected_feature=feature_i
               test_value=threshold
        alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
        predictions = pd.DataFrame(index=range(nsamples))
        predictions['values']=0
        predictions[train_X.iloc[:,selected_feature] < test_value] = 1
        bool_list=(predictions.values!=train_y.values).reshape((nsamples,))
        polarity=np.array([1 if x else -1 for x in bool_list])
        numerator=np.multiply(weights,np.exp(-alpha*polarity))
        normalizer=np.sum(numerator)
        weights=np.true_divide(numerator, normalizer)
        weakclassifiers.append(selected_feature)
        thresholds.append(test_value)
        del resampled_x
        del resampled_y
    return (weakclassifiers,thresholds)


if __name__=="__main__":
    #Importing dataset
    breast_cancer = load_breast_cancer()
    X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    y = pd.DataFrame(breast_cancer.target)

    #train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)
    (weakclassifiers, thresholds)=adaboost(X,y,5)
    print(weakclassifiers)
    print(thresholds)





