# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 00:06:34 2018

@author: SHALOM ALEXANDER
"""
    
def feature_split(Features):
    #60:20:20
    #train set
    m = len(Features)
    Feature_train = Features[0:int(0.6*m)]
    #test set
    mr = m - 0.6*m
    Feature_test = Features[int(0.6*m):int(0.6*m+mr*0.5)]
    #cross validation set
    Feature_cv = Features[int(0.6*m+mr*0.5):int(0.6*m+mr*0.5+mr*0.5)]
    return Feature_train,Feature_test,Feature_cv
