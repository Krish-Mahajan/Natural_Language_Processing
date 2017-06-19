#Email Classifier

##The Problem:
    - We have to build a model which will classify a given email to be spam or not spam  
    - Two sub models need to be built: Considering features as binary and continuous.  
    - Find the top 10 most spammy and least spammy words  

##The Approach/Formulation:
    - Fomrulation of dataset: We formulated the dataset to be considered as Term Frequency - Matrix Document  
      where in each email is the row and each feature is the word that exists in the document universe, and each cell represents the frequency of each word in that particular email. To save space, we maintain only the non-zero entries of this matrix for both the binary and continuous feature.  
    - Approach:
        - Naive - Bayes:
            - We begin reading the input data and forming in all three dictionaries:  
              a.) Words - Collection of all words and their frequencies  
              b.) Binary frequency - Records each word that appears in a given spam and not spam document.  
              c.) Continuous frequency - Records the number of times each word appears in a given spam and not spam document.  

            - Secondly, while adding all the words to dictionary, we performed a few string cleaning tasks:  
              a.) We removed the Punctuation  
              b.) We lowered the case of the string.  
              c.) We tried removing stop words and MIME headers, but have excluded this from the final programme since the accuracy was decreasing.

            - In, the next step, we calculate the priors and likelihood followed by testing:  
              a.) This means, we are trying to calculate the probabilty that a given email is spam or not spam given the list of words in that email.  
              b.) Say, an email contains 4 words, the probability of the mail being spam or not spam can thus be calculated as-
              Formula (For Spam) => P(S|W) = (P(W|S) * Prioir(S)) / (P(W|S) * Prior(S) + Pr(NS|W) * Prior(NS))  
              Formula (For Spam) => P(S=1|W1, W2, W3, W4) = P(W1|S=1) * P(W2|S=1) * P(W3|S=1) * P(W4|S=1) * Prior(S=1)  
              Formula (For Non Spam) => P(S|W) = (P(W|NS) * Prioir(NS)) / (P(W|S) * Prior(S) + Pr(NS|W) * Prior(NS))  
              Formula (For Non Spam) => P(S=0|W1, W2, W3, W4) = P(W1|S=0) * P(W2|S=0) * P(W3|S=0) * P(W4|S=0) * Prior(S=0)
              c.) We applied the above forumla to build a continious and binar model  
              d.) For Binary: We calculated the likelihood probability for binary as:
                  documents having the word, summing it over the total number of files present in that attribute.  
                P(W1|S=1) = Number of times W1 occurs in spam documents / Total number of spam documents
              e.) For Continuous: It was similar to the binary approach, herein we took the actualy frequency of the occurence P(W1|S=0) = Number of times W1 occurs in non-spam documents / Total number of non-spam documents of that word over the total occurence word.For Binary, the number of times W1 occurs is calculated by finding the number of documets in which W1 occurs.

              f.) This was the preparation model, after which we store this model in a pickle file for further use.  
              
              g.) While testing, we take each mail, apply same cleaning techniques to remove punctuation and change word to lower case and multiply all likelihoods together, followed by checking which class it lies in. (Note: Instead of multiplying (similar to that done in training). We then use the likelihood and prior probabilities calculated during training to find
              we simply took log and added all probabilies since the number obtained after multiplication was too small)
              P(S=1|W) and P(S=0|W) as per the formula given in point (b).
              If P(S=1|W) > P(S=0|W), we classify the mail as spam else as not spam. (Note: Instead of multiplying we simply took log of likelihood and prior and added all probabilities since the number obtained after multiplication was too small)  

        - Decision Trees:
            - Data Preparation and cleaning techniques are similar to Naive Bayes.  

            - Building the model:
                a.) We build the decision tree for both the binary and continuous feature.  
                b.) The idea is to achieve as pure subsets as possible without over training your model.  
                c.) The alogrithms basic idea is to split the current set into subsets, in such a way that you achieve as much less noise as possible.  
                d.) For the binary model, we take each word and divide the document set into two - one in which the word exists and other in which it does not exists.  
                which is the best split, selecting that and continuing the recursion until you get clean (or near clean) We then check the entropy of the parent node(initial document set) and the child node (two new sets created), the word returning the highest value for this entropy change is selected for splitting the dataset. subsets or if we hit a certain height.  
                We go on on splitting on each word in recursion until we run out of words/documents or we have clean subsets.  
                Clean subsets are identified as ones having zero entropy.
                
                e.)  For the continuous model, we take each word and find its average frequency in spam mails in which it occurs in the training dataset.  
                Say, if a word appears 400 times in a total of 40 spam mails, then on an average it appears 10 times in a spam mail.  
                We thus use this value as our splitting criteria.  
                We then divide the document set into two - one in which the word appears more than or equal to its average frequency and other in which it appears less number of times than its average frequency. We then check the entropy of the parent node(initial document set) and the child node (two new sets created), the word returning the highest value for this entropy change is selected for splitting the dataset. We go on on splitting on each word in recursion until we run out of words/documents or we have clean subsets.  

                f.) Once the tree is built, we save it in a pickle file for future use.  
                g.) In testing, we select each email, extract words, clean them after which we select each word from our model and check if it exists in the document, if it exists, it takes the word_exists branch otherwise it goes to and check if it exists in the document, if it exists, it takes the word_exists branch otherwise it goes to the word_not_Exists branch. This is done until we reach a leaf, and a decision made whether if it spam or not. the word_not_Exists branch for the binary model.  
                In case of continuous, it checks if the word occurs less than or greater than the average number of times it appeared in spam mails during training.  
                This is done until we reach a leaf, and a decision is made whether it is spam or not.  

                h.) Building the entire tree as per above approach was very time-consuming and was also prone to over-fitting and thus we did below changes-

                  (i) Instead of taking all the words, we find the words which are more probable to be in spam mails. We do this, by finding the ratio of sum of frequencies of a word in spam and sum of frequencies of that word in not spam. We then apply a threshold such that it removes words that appear a large number of times in both the spam and not spam mails since these words will provide little to no help in classifying mails.We thus restrict this ratio to be less than 0.45 to ensure we take words which contribute in filtering spam mails. We then also append words to this list, which appear only in spam mails with a high probability. Thus, we get our list of words which are most probable to be there in spam mails.  

                  (ii) We limit the depth to 10 since it takes less time to build the tree and also gives accuracy above 90%.  

##Results:
    - We gathered the following results after running our programme.  
    - Naive Bayes:
        - Continuous Model Accuracy: 98.36%
        - Confusion Matrix:  
                          Spam       Not Spam          Total  
           Spam           1156             29           1185  
       Not Spam             13           1356           1369  
          Total           1185           1385           2554  

        - Binary Model Accuracy: 98.2%  
        - Confustion Matrix:  
                          Spam       Not Spam          Total  
           Spam           1169             16           1185  
       Not Spam             30           1339           1369  
          Total           1185           1355           2554  

        - Time Taken for Training both models: 8.9s (on Burrow Server)  
        - Time Taken for Testing both models: 8.2s (on Burrow Server)  
    
    - Decision Tree:  
                          Spam       Not Spam          Total  
           Spam           1001            184           1185  
       Not Spam             53           1316           1369  
          Total           1185           1500           2554  

          Accuracy for decision tree for continuous feature:90.72%.


                          Spam       Not Spam          Total  
           Spam           1101             84           1185  
       Not Spam             47           1322           1369  
          Total           1185           1406           2554  
          Accuracy for decision tree for binary feature:94.87%  
  
        - Time Taken for Training both models: 193s (on Burrow Server)
        - Time Taken for Testing both models: 6.04s (on Burrow Server)

    - We had more success with the Naive Bayes model in terms of accuracy.  
    - In terms of time, training the Naive Bayes model takes relativiley lesser amounts of time but testing time is a little more, overall Naive Bayes did well.  


##Instructions to run:  
    - ```python spam.py mode technique dataset-dir model```  
    - To run training NB: ```python spam.py train bayes ./part1/train/ nb_model.p```  
    - To run testing NB: ```python spam.py test bayes ./part1/test/ nb_model.p```  
    - To run training DT: ```python spam.py train dt ./part1/train/ dt_model.p```  
    - To run testing DT: ```python spam.py test dt ./part1/test/ dt_model.p```  
    - You can give in any location of dataset, we will read the spam and non spam folders
      by ourselves, only giving the parent dir would do the trick  
    - Any name can be provided for model while training, be sure to use the same filename 
      while testing  

##References:
    - http://www.patricklamle.com/Tutorials/Decision%20tree%20python/tuto_decision%20tree.html  
    - http://codereview.stackexchange.com/questions/109089/id3-decision-tree-in-python  
    - https://www.youtube.com/watch?v=-dCtJjlEEgM  