
'''
- please go through readme.md for instruction on how to run the code and detailed analysis. 
I did following steps for the implementation :

- Read the Data and cleanned it through tokenizing ,regular expression,removing stopwords and selecting only words with minimum freq=3 
- Created a Vocabularly 
- Partitioned the data into label and unlabel as per FRACTION parameter
- Trained the data using semi-supervised naive bayes and five iteration of EM
- After the training two files are created MODEL-FILE.P & distinctive_words to store the trained model and file displaying top 10 words of each topic
- Now to predict the labels of testing data i am using softmax function on the trained model. 
- Finally i testing accuracy and confusion_matrix of the predicted label. 
- Results 
           fraction    Accuracy  
             0.0         10%
             0.01        42%
             0.5         79% 
             1           82%  

- For unsupervised(FRACTION=0) , i am assiging labels randomly. 
- To look at the confusion_matrix please check README>md
'''

import os
import pandas as pd  
import re 
import numpy as np
import math
import random
import snb  
import warnings 
import sys
import pickle 
import os
import pandas as pd  
import re 
import numpy as np
import math
import random

## Function to prepare training & testing data in tabular form
def read_data(path):
    """
    To read all the training data and return data as pandas data frame
    with three columns : id,text,label
    """
    data = []
    for topic in os.listdir(path):  
        if (topic != ".DS_Store"):
            new_path = "./"+path +"/" + topic 
            for document in os.listdir(new_path): 
                fo = open(new_path + "/" + document,'r')
                content = fo.read()
                fo.close 
                data.append({'id': str(topic + '-'+ document),'text':content,'label':topic})
    return pd.DataFrame(data)



def tokenizer(s):  
    """
    This function 'll clean each individual text and convert in 
    to tokens
    """
    s = re.sub("[^a-zA-Z]"," ",s)
    s = s.lower() 
    tokens=s.split(' ') 
    tokens = [t for t in tokens if len(t)>2] 
    tokens = [token for token in tokens if token not in stopwords] 
    return  tokens  

# def tokenize_data(train_data,test_data,min_freq):
def tokenize_test_data(data,word_index_map,min_freq):
    
    tokenized=[[]]    
    for index,row in data.iterrows(): 
        tokens = tokenizer(row['text']) 
        label = row['label']
        tokens.append(label)
        for token in tokens[:-1]: 
            if token not in word_index_map:
                tokens.remove(token)
        tokenized.append(tokens) 
        
    return tokenized

def tokenize_train_data(data,min_freq):
    tokenized = [[]]
    word_index_map = {} 
    current_index = 0 

    '''Reading all the documents(train + test) and tokenizing them and adding 
    to dictionary
    + Also making grand vocabularly V
    ''' 
            
    for index,row in data.iterrows(): 
        tokens = tokenizer(row['text']) 
        data.loc[index,'text']=" ".join(tokens)
        for token in tokens: 
            if token not in word_index_map:
                word_index_map[token] =[current_index ,1]
                current_index +=1 
            word_index_map[token][1] += 1  

        
    for index,row in data.iterrows(): 
        tokens = tokenizer(row['text']) 
        label = row['label']
        tokens.append(label)
        for token in tokens[:-1]: 
            if word_index_map[token][1]<=min_freq:
                tokens.remove(token)
        tokenized.append(tokens)  
                
    word_index_map_new={} 
    i=0
    for key in word_index_map.keys():
        if word_index_map[key][1]>min_freq:
            word_index_map_new[key]=i 
            i+=1
    word_index_map=word_index_map_new
                

    return data,word_index_map,tokenized


def tokens_to_vector(tokens,label_dict): 
    '''
    Vecotrizing particular token in test/train as per 
    Hashmap created Above          
    '''
    x = np.zeros(len(word_index_map)+1) 
    for t in tokens[:-1]: 
            j = word_index_map[t]
            x[j] +=1  
    x[-1]=label_dict[tokens[-1]]

    return x 


# def vectorize(train_data,test_data,word_index_map,training_tokenized,testing_tokenized):
def vectorize(data,word_index_map,tokenized,label_dict=None):  
    """
    Vectorizing all the text in training and testing
    """
    if not label_dict:
        label_dict={}
        i=0
        for label in data['label'].unique():
            label_dict[label]=i
            i+=1
    N = len(tokenized)-1
    data_vector = np.zeros((N,len(word_index_map)+1)) 
    i=0
    for tokens in tokenized[1:]:
        xy = tokens_to_vector(tokens,label_dict)   
        data_vector[i,:] = xy 
        i +=1    
 
    return data_vector,label_dict



def coin_flip(fraction):
    return (random.random()>=1-fraction)


def partition_label_unlabeled(train_data_vector,fraction):
    """
    To segregate training data into labeled and non labeled based on fraction
    """ 
    if fraction ==0:
        fraction = 0.0001

    for index in range(len(train_data_vector)):
        if (not coin_flip(fraction)):train_data_vector[index][-1]=-1
    train_data_vector_labeled = train_data_vector[train_data_vector[:,-1]!=-1]
    train_data_vector_unlabeled = train_data_vector[train_data_vector[:,-1]==-1]
    
    return train_data_vector_labeled,train_data_vector_unlabeled


def term_document(data): 
    """
    td: term-document matrix V x D
    """
    td = data.T[:-1,:] 
    return td 


def delta(data,td):
    """
    delta: D x T matrix
    where delta_train(d,c) = 1 if document d belongs to class c
    """
    total_classes = len(np.unique(data[:,-1]))
    total_documents = td.shape[1]
    delta_train = np.zeros((total_documents,total_classes))  
    
    label=data[:,-1]  

    #Filling Delta
    for row in range(len(label)) :
        delta_train[row,label[row]]=1
        
    return delta_train



####Testing #####  

if __name__=="__main__": 
    warnings.filterwarnings('ignore')
    # stopwords = set(w.rstrip() for w in open('stopwords.txt'))  
    stopwords ={'about', 'above', 'across', 'after', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 
    'always', 'among', 'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are', 'area', 'areas', 'around', 'as', 
    'ask', 'asked', 'asking', 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be', 'became', 'because', 'become', 'becomes',
    'been', 'before', 'began', 'behind', 'being', 'beings', 'best', 'better', 'between', 'big', 'both', 'but', 'by', 'c', 'came', 'can', 
    'cannot', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 'come', 'could', 'd', 'did', 'differ', 'different', 'differently', 
    'do', 'does', 'done', 'down', 'downed', 'downing', 'downs', 'during', 'e', 'each', 'early', 'either', 'end', 'ended', 'ending', 'ends', 
    'enough', 'even', 'evenly', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'f', 'face', 'faces', 'fact', 'facts', 
    'far', 'felt', 'few', 'find', 'finds', 'first', 'for', 'four', 'from', 'full', 'fully', 'further', 'furthered', 'furthering', 'furthers', 
    'g', 'gave', 'general', 'generally', 'get', 'gets', 'give', 'given', 'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 'greater', 
    'greatest', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', 'has', 'have', 'having', 'he', 'hen', 'her', 'here', 'herself', 'high', 
    'higher', 'highest', 'him', 'himself', 'his', 'how', 'howevhowevhoif', 'important', 'in', 'interest', 'interested', 'interesting', 
    'interests', 'into', 'is', 'it', 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kind', 'knew', 'know', 'known', 'knows', 'l', 
    'large', 'largely', 'last', 'later', 'latest', 'least', 'less', 'let', 'lets', 'like', 'likely', 'long', 'longer', 'longest', 'm', 
    'made', 'make', 'making', 'man', 'many', 'may', 'me', 'member', 'members', 'men', 'might', 'more', 'most', 'mostly', 'mr', 'mrs', 'much',
    'must', 'my', 'myself', 'n', 'need', 'needed', 'needing', 'needs', 'never', 'new', 'newer', 'newest', 'next', 'nnnnnsary', 'no', 
    'nobody', 'non', 'noone', 'not', 'nothing', 'now', 'nowhere', 'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 
    'oldest', 'on', 'once', 'one', 'only', 'open', 'opened', 'opening', 'opens', 'or', 'order', 'ordered', 'ordering', 'orders', 
    'other', 'others', 'our', 'out', 'over', 'p', 'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 
    'point', 'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 'presents', 'problem', 'problems', 
    'put', 'puts', 'q', 'quite', 'r', 'rather', 'really', 'right', 'room', 'rooms', 's', 'said', 'same', 'saw', 'say', 'says', 
    'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 'seems', 'sees', 'several', 'shall', 'she', 'should', 'show', 'showed', 
    'showing', 'shows', 'side', 'sides', 'since', 'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'someone', 'something', 
    'somewhere', 'state', 'states', 'still', 'such', 'sure', 't', 'take', 'taken', 'than', 'that', 'the', 'their', 'them', 'there', 
    'therefore', 'these', 'they', 'thing', 'things', 'think', 'thinks', 'those', 'though', 'thought', 'thoughts', 'three', 'through', 
    'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'turn', 'turned', 'turning', 'turns', 'two', 'u', 'under', 'until', 'up', 
    'upon', 'us', 'use', 'used', 'uses', 'v', 'very', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', 'way', 'ways', 'we', 'well', 
    'wells', 'went', 'were', 'what', 'when', 'where', 'whether', 'which', 'while', 'who', 'whole', 'whose', 'why', 'will', 'with', 'within', 
    'without', 'work', 'worked', 'working', 'works', 'would', 'x', 'y', 'year', 'years', 'yet', 'younger', 'youngest', 'your', 'yours', 
    'yyyyyyyung', 'z','you','this','people','com','subject','lines','edu','lines','writes','article'}
    # stopwords = {}

    mode = sys.argv[1] 
    dataset_directory = sys.argv[2] 
    model_file = sys.argv[3]
    fraction = float(sys.argv[4]) 


    if(mode=="train"): 
        print("Reading Training Data...")
        train_data = read_data(dataset_directory)  


        print("Tokinizing Train Data and Creating Vocabularly")
        train_data,word_index_map,training_tokenized=tokenize_train_data(train_data,3)   


        print("Vectorizing Training Data...")
        train_data_vector,label_dict = vectorize(train_data,word_index_map,training_tokenized)

        print("Splitting training Data into label and unlabeled as per ",fraction,"probability of looking at each label")
        train_data_vector_label,train_data_vector_unlabel = partition_label_unlabeled(train_data_vector,fraction)


        print("Making term document matrix of Training label Data and Training unlabel Data")
        tf_train_label = term_document(train_data_vector_label) 
        tf_train_unlabel = term_document(train_data_vector_unlabel) 

        print("Making Delta on training label")
        delta_train_label = delta(train_data_vector_label,tf_train_label) 

        print("Data Prepared")  

        nb=snb.NaiveBayes(vocab=word_index_map,label_dict=label_dict)

        print("Training EM naive Bayes Model...") 
        nb.train_semi(tf_train_label,delta_train_label,tf_train_unlabel)  


        print("Making distinctive_words for top 10 words in each topic")
        nb.top_10_words()


        print("Saving Trained Model..")
        f = open("model-file.p","w")
        pickle.dump(nb,f)
        f.close()



    if(mode=="test"): 

        
        print("opening Saved semi supervised naieve bayes model") 
        ff = open("model-file.p","r")
        nb=pickle.load(ff)
        ff.close() 

        print("Reading Testing Data...")
        test_data=read_data(dataset_directory)   

        print("Tokinizing Test Data")
        testing_tokenized =tokenize_test_data(test_data,nb.vocab,3)    


        print("Vectorizing Test Data...")
        word_index_map=nb.vocab
        label_dict = nb.label_dict
        test_data_vector,label_dict= vectorize(test_data,word_index_map,testing_tokenized,label_dict)

        print("Making term document matrix of testing Data")
        tf_test = term_document(test_data_vector) 


        print("Making Delta on testing label")
        delta_test = delta(test_data_vector,tf_test) 

        
    
        print("Predicting test Data")
        predict=nb.predict_proba_all(tf_test)
        print("Accuracy is",nb.accuracy(predict,delta_test))

        print("Confusion Matrix:")
        print(nb.confusion_matrix(delta_test,predict,nb.label_dict))
        
     