###################################
# CS B551 Fall 2016, Assignment #3
#
# Your names and user ids: krimahaj,songyan,vranga
#
# (Based on skeleton code by D. Crandall)
#
#
'''
First using training Data calculated transition matrix,emission matrix, iniital probability matrix

# for part1 simply took arg max P(S=s|W)  as directed 

# for part2 took p(Q=q)*P(Q_t+1=q_t+1|Q_t=q_t)*P(O_t|Q_t) using viterbi algorithm & Backtracking  
  mainted two matrices delta to store max probabilities and phi to keep track of states to be used in Backtracking

# For part3 took p(Q=q)*P(Q_t+1=q_t+1|Q_t=q_t,Q_t-1=q_t-1)*P(O_t|Q_t) 
  calculated trigram transition probabilities to take into account previous two states 
  and then viterbi algorithm as in part two to get most probable states
'''

import random
import math
import numpy as np

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver: 

    def __init__(self):
        self.M = None # number of hidden states
        self.pi=None #initial
        self.A =None #Transition
        self.B =None #emission
        self.word2idx =None #wordDictionary
        self.tag2idx= None  #TagDcitionary
        self.V=None



    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label): 

            BBH= self.B/self.B.sum(axis=1, keepdims=True) 
            prob=1
            s=[]
            for word in sentence:
                if word not in self.word2idx: 
                    s.append(self.V)
                else:
                    s.append(self.word2idx[word])  

            l=[]
            for tag in label: l.append(self.tag2idx[tag])  

            prob=np.log(self.pi[l[0]])
            for i in range(len(s[:-1])): prob = prob + np.log(BBH[l[i],s[i]]) + np.log(self.A[l[i],l[i+1]])
            prob =prob + np.log(BBH[l[-1],s[-1]])
            return round(prob,2)

    
   


    # Do the training!
    def train(self, data):
        word2idx,tag2idx = {},{}  #Word Vocabulary with word as key and word_idx as value ,#Tag Vocabulary with Tag as key and Tag_idx as value
        word_idx,tag_idx = 0,0  #counter for words , tags
        Xtrain,Ytrain = [],[]    #for storing overall Sentence & Tag sequences
        currentX,currentY = [],[]  #to store current word sequences (observed Data) & tag sequences (unobserved Data)
        for sentence in data:
            for word,tag in zip(sentence[0],sentence[1]): 
                if word not in word2idx:
                        word2idx[word] = word_idx
                        word_idx += 1
                currentX.append(word2idx[word])

                if tag not in tag2idx: 
                    tag2idx[tag] = tag_idx
                    tag_idx += 1
                currentY.append(tag2idx[tag])
            Xtrain.append(currentX)
            Ytrain.append(currentY)
            currentX = []
            currentY = []

        V = len(word2idx) + 1 #Vocabulary Size

        # find hidden state, transition matrix and pi
        M = max(max(y) for y in Ytrain) + 1 #find total number of hiddent states
        A = np.ones((M, M))*(10e-60) # add-one smoothing
        pi = np.zeros(M) #initial Matrix

        for y in Ytrain:
            pi[y[0]] += 1
            for i in xrange(len(y)-1):
                A[y[i], y[i+1]] += 1

        # turn it into a probability matrix
        A /= A.sum(axis=1, keepdims=True) #state transition matrix
        pi /= pi.sum() #initial matrix

        # find the observation matrix
        B = np.ones((M, V+1))*(10e-60) # add smoothing
        for x, y in zip(Xtrain, Ytrain):
            for xi, yi in zip(x, y):
                B[yi, xi] += 1 


        self.M=M
        self.pi = pi
        self.A = A
        self.B = B 
        self.word2idx = word2idx
        self.tag2idx= tag2idx  
        self.V=V


    # Functions for each algorithm.
    #
    def simplified(self, sentence): 

        BBS = self.B/self.B.sum(axis=0, keepdims=True) 
        # turn it into a probability matrix

        x=[]
        for word in sentence:
            if word not in self.word2idx: 
                x.append(self.V)
            else:
                x.append(self.word2idx[word]) 
        T = len(x)
        states =[]
        probs = []
        for word in x:
            probs.append(np.max(BBS[:,word]))
            states.append(np.argmax(BBS[:,word]))

        tag_states=[]
        for i in range(len(states)):
            for tag,tag_idx in self.tag2idx.iteritems():
                if tag_idx==states[i]: tag_states.append(tag)
 
        return [[tag_states],[probs]]


    def hmm(self, sentence): 

        BBH= self.B/self.B.sum(axis=1, keepdims=True) 

        # turn it into a probability matrix
        self.A /= self.A.sum(axis=1, keepdims=True) #state transition matrix
        self.pi /= self.pi.sum() #emission matrix

        # returns the most likely state sequence given observed sequence x
        # using the Viterbi algorithm 
        x=[]
        for word in sentence:
            if word not in self.word2idx: 
                x.append(self.V)
            else:
                x.append(self.word2idx[word])

        T = len(x)
        delta = np.zeros((T, self.M))
        psi = np.zeros((T, self.M))
        delta[0] = np.log(self.pi) + np.log(BBH[:,x[0]])
        for t in xrange(1, T):
            for j in xrange(self.M):
                delta[t,j] = np.max(delta[t-1] + np.log(self.A[:,j])) + np.log(BBH[j, x[t]])
                psi[t,j] = np.argmax(delta[t-1] + np.log(self.A[:,j]))

        # backtrack
        states = np.zeros(T, dtype=np.int32)
        states[T-1] = np.argmax(delta[T-1])
        for t in xrange(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        tag_states=[]
        for i in range(len(states)):
            for tag,tag_idx in self.tag2idx.iteritems():
                if tag_idx==states[i]: tag_states.append(tag)
            
        return [[tag_states],[]]


    def complex(self, sentence):  
        BBH= self.B/self.B.sum(axis=1, keepdims=True)  

        BBH[:,self.V]=0
        BBH[1:,self.V]=1

        self.A /= self.A.sum(axis=1, keepdims=True) #state transition matrix
        self.pi /= self.pi.sum() #emission matrix

        # returns the most likely state sequence given observed sequence x
        # using the Viterbi algorithm 
        x=[]
        for word in sentence:
            if word not in self.word2idx: 
                x.append(self.V)
            else:
                x.append(self.word2idx[word])

        T = len(x)
        delta = np.zeros((T, self.M))
        psi = np.zeros((T, self.M))
        delta[0] = np.log(self.pi) + np.log(BBH[:,x[0]])
        for t in xrange(1, T):
            for j in xrange(self.M):
                delta[t,j] = np.max(delta[t-1] + np.log(self.A[:,j])) + np.log(BBH[j, x[t]])
                psi[t,j] = np.argmax(delta[t-1] + np.log(self.A[:,j]))

        # backtrack
        states = np.zeros(T, dtype=np.int32)
        states[T-1] = np.argmax(delta[T-1])
        for t in xrange(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        tag_states=[]
        for i in range(len(states)):
            for tag,tag_idx in self.tag2idx.iteritems():
                if tag_idx==states[i]: tag_states.append(tag) 

        probs = []
        for word in x:
            probs.append(np.max(BBH[:,word]))


            
        return [[tag_states],[probs]]



    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for simplified() and complex() and is the marginal probability for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM":
            return self.hmm(sentence)
        elif algo == "Complex":
            return self.complex(sentence)
        else:
            print "Unknown algo!"

