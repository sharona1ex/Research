# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 19:03:38 2018

@author: SHALOM ALEXANDER
"""
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix

class multi_svm:
    
    def __init__(self,K,Kernel,Gamma,c):
        self.clf_arr = []
        for i in range(K):
            self.clf_arr.append(svm.SVC(kernel = Kernel,gamma = Gamma,C = c))
    
    def train(self,Feature,Label):
        i = 0
        for clf in self.clf_arr:
            clf.fit(Feature,Label[:,i])
            i = i + 1
            
    def predict(self,Feature,Label):
        i = 0
        net_av_error = 0
        av_error = []
        for clf in self.clf_arr:
            pred = clf.predict(Feature)
            mat = confusion_matrix(Label[:,i],pred)
            num_of_examples = len(Feature)
            av_error.append((num_of_examples - sum([mat[i][i] for i in range(len(mat))]))/num_of_examples)
            i = i + 1
        net_av_error = sum(av_error)
        return net_av_error,av_error
