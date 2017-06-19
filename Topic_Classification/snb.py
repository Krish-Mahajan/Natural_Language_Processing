"""
Text Classification using Naive Bayes on Expectation Maximization Principle
w: word
x: document
T: topic/class
V: vocabulary 
M: number of documents
T: number of classes
""" 

import numpy as np
import pandas as pd


class NaiveBayes(object):

    def __init__(self, vocab=None,label_dict=None):

        self.p_w_c = None
        self.p_c = None
        self.vocab=vocab
        self.label_dict=label_dict   

        # to normalize p(c)
    def normalize_p_c(self,p_c):
        M = len(p_c)
        denom = M + np.sum(p_c)
        p_c += 1.0
        p_c /= denom
        p_c /=np.sum(p_c)
        
        # to normalize p(w|c)
    def normalize_p_w_c(self,p_w_c):
        V, X = p_w_c.shape
        denoms = V + np.sum(p_w_c, axis=0)
        p_w_c += 1.0
        p_w_c /= denoms[np.newaxis,:]  


    def top_10_words(self): 
        f = open("distinctive_words.txt","w")
        for i in range(self.p_w_c.shape[1]):
            column = []
            for x,y in np.ndenumerate(self.p_w_c[:,i]):
                column.append((x[0],y)) 
            top_10 = sorted(column, key=lambda x: x[1])[-10:] 
            top_10.reverse()
            f.write("Top words for topic  "+ self.label_dict.keys()[self.label_dict.values().index(i)].upper())
            f.write("\n")
            for word in top_10:
                f.write(self.vocab.keys()[self.vocab.values().index(word[0])])
                f.write("\n") 
            f.write("\n")
        f.close()


        # to train naive bayes model
    def train(self, td, delta, normalize=True):

        X_, M = delta.shape
        V, X = td.shape

        # P(c)
        self.p_c = np.sum(delta, axis=0)

        # P(w|c)
        self.p_w_c = np.zeros((V,M), dtype=np.double)

        for w,d in zip(*td.nonzero()):
            self.p_w_c[w,:] += td[w,d] * delta[d,:]


        if normalize:
            self.normalize_p_c(self.p_c)
            self.normalize_p_w_c(self.p_w_c) 

        #to train naive bayes when model is semi supervised
    def train_semi(self, td, delta, tdu, maxiter=5):
        X_, M = delta.shape
        V, X = td.shape

        # compute counts for labeled data once for all
        self.train(td, delta, normalize=False)
        p_c_l = np.array(self.p_c, copy=True)
        p_w_c_l = np.array(self.p_w_c, copy=True)

        # normalize to get initial classifier
        self.normalize_p_c(self.p_c)
        self.normalize_p_w_c(self.p_w_c)

        for iteration in range(1, maxiter+1):
            # E-step: 
            print("iteration no ",iteration," out of total",maxiter)
            delta_u = self.predict_proba_all(tdu)

            # M-step: 
            self.train(tdu, delta_u, normalize=False)
            self.p_c += p_c_l
            self.p_w_c += p_w_c_l
            self.normalize_p_c(self.p_c)
            self.normalize_p_w_c(self.p_w_c)

    def p_x_c_log_all(self, td):
        M = len(self.p_c)
        V, X = td.shape
        p_x_c_log = np.zeros((X,M), np.double)
        p_w_c_log = np.log(self.p_w_c)

        for w,d in zip(*td.nonzero()):
            p_x_c_log[d,:] += p_w_c_log[w,:] * td[w,d]

        return p_x_c_log

    def max_prob(self,loga, k=-np.inf, out=None):
        if out is None: out = np.empty_like(loga).astype(np.double)
        m = np.max(loga)
        logam = loga - m
        sup = logam > k
        inf = np.logical_not(sup)
        out[sup] = np.exp(logam[sup])
        out[inf] = 0.0
        out /= np.sum(out)
        return out

    def predict_proba_all(self, td):
        V, X = td.shape
        p_x_c_log = self.p_x_c_log_all(td)
        p_x_c_log += np.log(self.p_c)[np.newaxis,:]
        for d in range(X):
            self.max_prob(p_x_c_log[d,:], k=-10, out=p_x_c_log[d,:]) 

        return p_x_c_log

        
    def accuracy(self,result,delta_test):
        i=0
        for predict,actual in zip(result,delta_test):
            if np.argmax(predict)==np.argmax(actual): i +=1 
        accuracy = i*100.0/len(delta_test)
        return accuracy  
    
    def confusion_matrix(self,delta_test,predict,label_dict):
        y_actual =[np.argmax(row) for row in delta_test]
        y_pred=[np.argmax(row) for row in predict]

        for row in range(len(y_actual)):
            for key,value in label_dict.iteritems():
                if value==y_actual[row]: y_actual[row]=key

        for row in range(len(y_pred)):
            for key,value in label_dict.iteritems():
                if value==y_pred[row]: y_pred[row]=key 

        y_actual = pd.Series(y_actual,name='Actual') 
        y_pred = pd.Series(y_pred,name='predicted')

        df_confusion = pd.crosstab(y_actual, y_pred)
        return df_confusion

