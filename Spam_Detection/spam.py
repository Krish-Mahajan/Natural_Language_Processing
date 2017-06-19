import sys
from os import listdir
from os.path import isfile, join
import string
import json
import math
import operator
from operator import getitem
from collections import OrderedDict
from operator import itemgetter
import pickle
import six
import time

debug = 0

"""

DOCUMENTATION GOES HERE:

- The Problem:
    - We have to build a model which will classify a given email to be spam or not spam
    - Two sub models need to be built: Considering features as binary and continuous.
    - Find the top 10 most spammy and least spammy words

- The Approach/Formulation:
    - Formulation of dataset: We formulated the dataset to be considered as Term Frequency - Matrix Document
      where in each email is the row and each feature is the word that exists in the document universe, and
      each cell represents the frequency of each word in that particular email. To save space, we maintain only the non-zero entries of this matrix for both the binary and continuous feature.
    - Approach:
        - Naive - Bayes:
            - We begin reading the input data and forming in all three dictionaries:
              a.) Words - Collection of all words and their frequencies
			  b.) Binary frequency - Records each word that appears in a given spam and not spam document
			  c.) Continuous frequency - Records the number of times each word appears in a given spam and not spam document


            - Secondly, while adding all the words to dictionary, we performed a few string cleaning tasks:
              a.) We removed the Punctuation
              b.) We lowered the case of the string.
              c.) We tried removing stop words and MIME headers, but have excluded this from the final programme
                  since the accuracy was decreasing.

            - In, the next step, we calculate the priors and likelihood followed by testing:
              a.) This means, we are trying to calculate the probabilty that a given email is spam or not spam given the list of words in that email
              b.) Say, an email contains 4 words, the probability of the mail being spam or not spam can thus be calculated as-
				  Formula (For Spam) => P(S=1|W1, W2, W3, W4) = P(W1|S=1) * P(W2|S=1) * P(W3|S=1) * P(W4|S=1) * Prior(S=1)
                  Formula (For Non Spam) => P(S=0|W1, W2, W3, W4) = P(W1|S=0) * P(W2|S=0) * P(W3|S=0) * P(W4|S=0) * Prior(S=0)
              c.) We applied the above forumla to build a continuous and binary model for Bayes classifier
              d.) For Binary: We calculated the likelihood probability for binary as:
				  P(W1|S=1) = Number of times W1 occurs in spam documents / Total number of spam documents
				  P(W1|S=0) = Number of times W1 occurs in non-spam documents / Total number of non-spam documents
				  For Binary, the number of times W1 occurs is calculated by finding the number of documets in which W1 occurs,
				  i.e., at max 1 occurrence per document.
              e.) For Continuous: We calculated the likelihood probability for continuous as:
			      P(W1|S=1) = Number of times W1 occurs in spam documents / Total number of words in spam documents
				  P(W1|S=0) = Number of times W1 occurs in non-spam documents / Total number of words in non-spam documents
				  For Continuous, the number of times W1 occurs is calculated by summing up the frequency of W1 in each of the spam documents where it occurs.
              f.) This was the preparation model, after which we store this model in a pickle file for further use
              g.) While testing, we take each mail, apply same cleaning techniques to remove punctuation and change word to lower case
				  (similar to that done in training). We then use the likelihood and prior probabilities calculated during training to find
				  P(S=1|W) and P(S=0|W) as per the formula given in point (b).
				  If P(S=1|W) > P(S=0|W), we classify the mail as spam else as not spam. (Note: Instead of multiplying
                  we simply took log of likelihood and prior and added all probabilities since the number obtained after multiplication was too small)

        - Decision Trees:
            - Data Preparation and cleaning techniques are similar to Naive Bayes.

            - Building the model:
                a.) We build the decision tree for both the binary and continuous feature.
                b.) The idea is to achieve as pure subsets as possible without over training your model.
                c.) The alogrithms basic idea is to split the current set into subsets, in such a way that you achieve as much
                    less noise as possible.
                d.) We consider our splitting attribute as words and splitting criterion as change in entropy of the parent node and child node (information gain)
                e.) First step is to select the best possible attribute value resulting in near clean subsets.
				d.) For the binary model, we take each word and divide the document set into two - one in which the word exists and other in which it does not exists.
				    We then check the entropy of the parent node(initial document set) and the child node (two new sets created), the word returning the highest value for this entropy change is selected for splitting the dataset.
					We go on on splitting on each word in recursion until we run out of words/documents or we have clean subsets.
					Clean subsets are identified as ones having zero entropy.
				e.) For the continuous model, we take each word and find its average frequency in spam mails in which it occurs in the training dataset.
				    Say, if a word appears 400 times in a total of 40 spam mails, then on an average it appears 10 times in a spam mail.
					We thus use this value as our splitting criteria.
				    We then divide the document set into two -
				    one in which the word appears more than or equal to its average frequency and other in
					which it appears less number of times than its average frequency.
				    We then check the entropy of the parent node(initial document set) and the child node (two new sets created), the word returning the highest value for this entropy change is selected for splitting the dataset.
					We go on on splitting on each word in recursion until we run out of words/documents or we have clean subsets.
					Clean subsets are identified as ones having zero entropy.
                f.) Once the tree is built, we save it in a pickle file for future use.
                g.) In testing, we select each email, extract words, clean them after which we select each word from our model
                    and check if it exists in the document, if it exists, it takes the word_exists branch otherwise it goes to
                    the word_not_Exists branch for the binary model.
					In case of continuous, it checks if the word occurs less than or greater than the average number
					of times it appeared in spam mails during training.
					This is done until we reach a leaf, and a decision is made whether it is spam or not.
				h.) Building the entire tree as per above approach was very time-consuming and was also prone to over-fitting and
				    thus we did below changes-
					(i) Instead of taking all the words, we find the words which are more probable to be in spam mails.
					We do this, by finding the ratio of sum of frequencies of a word in spam and sum of frequencies of that word in not spam.
					We then apply a threshold such that it removes words that appear a large number of times in both the spam and not spam mails
					since these words will provide little to no help in classifying mails.We thus restrict this ratio to be less than 0.45 to
					ensure we take words which contribute in filtering spam mails.
					We then also append words to this list, which appear only in spam mails with a high probability.
					Thus, we get our list of words which are most probable to be there in spam mails.
					(ii) We limit the depth to 10 since it takes less time to build the tree and also gives accuracy above 90%.

- Results:
    - We gathered the following results after running our programme
    - Naive Bayes:
        - Continuous Model Accuracy: 98.36%
        - Confusion Matrix:
                          Spam       Not Spam          Total
           Spam           1156             29           1185
       Not Spam             13           1356           1369
          Total           1185           1385           2554

        - Binary Model Accuracy: 98.2%
        - Time taken: 8.3s (on Burrow Server)
        - Confusion Matrix:
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
          Accuracy for decision tree for continuous feature:90.72

                          Spam       Not Spam          Total
           Spam           1101             84           1185
       Not Spam             47           1322           1369
          Total           1185           1406           2554
          Accuracy for decision tree for binary feature:94.87

        - Time Taken for Training both models: 193s (on Burrow Server)
        - Time Taken for Testing both models: 6.04s (on Burrow Server)

    - We had more success with the Naive Bayes model in terms of accuracy.
    - In terms of time, training the Naive Bayes model takes relatively lesser amounts of time
      but testing time is a little more, overall Naive Bayes did well.

- Instructions to run:
    - python spam.py mode technique dataset-dir model
    - To run training NB: python spam.py train bayes ./part1/train/ nb_model.p
    - To run testing NB: python spam.py test bayes ./part1/test/ nb_model.p
    - To run training DT: python spam.py train dt ./part1/train/ dt_model.p
    - To run testing DT: python spam.py test dt ./part1/test/ dt_model.p
    - You can give in any location of dataset, we will read the spam and non spam folders
      by ourselves, only giving the parent dir would do the trick
    - Any name can be provided for model while training, be sure to use the same filename 
      while testing

- References:
    - http://www.patricklamle.com/Tutorials/Decision%20tree%20python/tuto_decision%20tree.html
    - http://codereview.stackexchange.com/questions/109089/id3-decision-tree-in-python
    - https://www.youtube.com/watch?v=-dCtJjlEEgM
"""

is_python3 = sys.version_info.major == 3
if is_python3:
    translator = str.maketrans({key: None for key in string.punctuation})
    unicode = str

dir_type = ['spam', 'notspam']
words = {}
doc_freq = {}
doc_binary = {}
frequency = {}
binary = {}
words_type = {}
docs_type = {}
prob_type = {}
cont_likelihood = {}
bin_likelihood = {}
visited_words = []
entropy = {}
train_file_list = {}
reduced_words = []

def get_list_of_files(path):
    files = {}
    for each in dir_type:
        filelist = [f for f in listdir(path + each + '/''') if isfile(join(path + each + '/''', f))]
        files[each] = filelist
    # print(files.keys())
    return files


def read_train(path):
    # path = './part1/train/'
    for each_type in dir_type:
        words_type[each_type] = 0
        doc_freq[each_type] = {}
        doc_binary[each_type] = {}
        filelist = train_file_list[each_type]
        for each_file in filelist:

            if (is_python3):
                with open(path + each_type + '/''' + each_file, encoding='utf-8', errors='ignore') as f:
                    word_list = f.read().split()
            else:
                with open(path + each_type + '/''' + each_file) as f:
                    word_list = f.read().split()

                for each_word in word_list:
                    # Need to check if this will remove hyphen within a word
                    each_word = clean_string(each_word)
                    if len(each_word) > 0:
                        words_type[each_type] += 1
                        # Adding words to our vocabulary
                        if each_word in words:
                            words[each_word] += 1
                        else:
                            words[each_word] = 1

                        if each_word not in doc_freq[each_type]:
                            doc_freq[each_type][each_word] = {}
                            doc_binary[each_type][each_word] = {}

                        # Creating histogram of words and binary feature for each document
                        if each_file in doc_freq[each_type][each_word]:
                            doc_freq[each_type][each_word][each_file] += 1
                        else:
                            doc_freq[each_type][each_word][each_file] = 1
                            doc_binary[each_type][each_word][each_file] = 1

            f.close()

def clean_string(s):
    if (is_python3):
        s = s.translate(translator)
    else:
        s = unicode(s.translate(None, string.punctuation), errors='ignore')
    return s.lower()


def file_dump(words, filename='output'):
    with open(filename + '.json', 'w') as f:
        f.write(json.dumps(words, indent=4))
    # print("done")
    f.close()


# -------------------------------Bayes Training----------------------------------------

def get_prior():
    total_docs = 0
    for each_type in dir_type:
        docs_type[each_type] = len(train_file_list[each_type])
        total_docs += docs_type[each_type]
    for each_type in dir_type:
        prob_type[each_type] = docs_type[each_type] / float(total_docs)


def get_likelihood(likelihood, feature, total_deno):
    for each_type in dir_type:
        likelihood[each_type] = {}
        for each_word in words:
            if each_word in feature[each_type]:
                likelihood[each_type][each_word] = sum(feature[each_type][each_word].itervalues()) / float(total_deno[each_type])
            else:
                likelihood[each_type][each_word] = 0.5/float(total_deno[each_type])


def train_bayes(path):
    read_train(path)
    get_prior()
    get_likelihood(cont_likelihood, doc_freq, words_type)
    get_likelihood(bin_likelihood, doc_binary, docs_type)


# -------------------------------Decision Tree Classifier Training----------------------------------------

class node:
    def __init__(self, word='', frequency=-1, exists=None, not_exists=None, label='NA'):
        self.word = word
        self.frequency = frequency
        self.exists = exists
        self.not_exists = not_exists
        self.label = label


def get_type_count(file_list):
    spam_count = 0
    notspam_count = 0
    type_count = []
    for each_file in file_list:
        if each_file in train_file_list['spam']:
            spam_count += 1
        else:
            notspam_count += 1
    type_count.append(spam_count)
    type_count.append(notspam_count)

    return type_count


def calc_entropy(file_list):
    total_files = len(file_list)
    node_features = get_type_count(file_list)
    if total_files == 0:
        node_features.append(0)
    else:
        if node_features[0] > 0: #spam_count
            entropy_spam = -(node_features[0]/float(total_files)) * math.log(node_features[0]/float(total_files))
        else:
            entropy_spam = 0
        if node_features[1] > 0: #spam_count
            entropy_notspam = -(node_features[1]/float(total_files)) * math.log(node_features[1]/float(total_files))
        else:
            entropy_notspam = 0
        node_features.append(entropy_spam + entropy_notspam)

    return node_features


# Below function splits the data and returns the entropy
def get_split_entropy(word, file_list, feature):
    docs_for_word = []
    spam_docs = []
    notspam_docs = []
    avg_freq = 0
    if word in doc_binary['spam']:
        spam_docs = doc_binary['spam'][word].keys()
    if word in doc_binary['notspam']:
        notspam_docs = doc_binary['notspam'][word].keys()
    docs_for_word = spam_docs + notspam_docs
    exists = list(set(docs_for_word) & set(file_list))
    not_exists = list(set(file_list) - set(exists))
    total = len(exists) + len(not_exists)

    if feature != 'binary':
        avg_freq = math.floor(
            sum(doc_freq['spam'][word].itervalues()) / float(sum(doc_binary['spam'][word].itervalues())))
        for each_file in exists:
            if each_file in spam_docs and doc_freq['spam'][word][each_file] < avg_freq:
                exists.remove(each_file)
                not_exists.append(each_file)
            if each_file in notspam_docs and doc_freq['notspam'][word][each_file] < avg_freq:
                exists.remove(each_file)
                not_exists.append(each_file)

    if debug == 1:
        print (len(exists), len(not_exists), total)

    exists_data = calc_entropy(exists)
    not_exists_data = calc_entropy(not_exists)

    exists_data[2] *= (len(exists) / float(total))
    not_exists_data[2] *= (len(not_exists) / float(total))
    entropy_word = exists_data[2] + not_exists_data[2]

    return entropy_word, avg_freq, exists, not_exists, exists_data, not_exists_data


def build_dtree(file_list, feature, max_depth):
    # print "Total files in this iteration: " + str(len(file_list))
    parent_entropy = calc_entropy(file_list)[2]
    # print "Parent entropy:" + str(parent_entropy)
    # min_entropy = 2
    max_gain = 0
    best_word = ''
    word_freq = -1
    exists_tree = []
    notexists_tree = []
    exists_tree_data = []
    notexists_tree_data = []

    if len(reduced_words) != len(visited_words) and max_depth > 0:
        # max_depth -= 1
        for each_word in reduced_words:
            # To reduce the list of words
            if words[each_word] <= 100:
                visited_words.append(each_word)
            # To ensure we do not split on the same attribute
            if each_word not in visited_words:
                # print "Next word:" + each_word
                entropy, freq, exists, notexists, exists_data, notexists_data = get_split_entropy(each_word, file_list, feature)
                if max_gain < (parent_entropy - entropy) and len(exists) > 0 and len(notexists) > 0:
                    max_gain = parent_entropy - entropy
                    best_word = each_word
                    word_freq = freq
                    exists_tree = exists
                    notexists_tree = notexists
                    exists_tree_data = exists_data
                    notexists_tree_data = notexists_data
                    # print "Found new best word:"
                    # print best_word, max_gain
    if debug == 1:
        print ("Final word, frequency, gain and max depth:")
        print (best_word, word_freq, max_gain, max_depth)

    if max_gain > 0:
        visited_words.append(best_word)

        # Check for sub tree entropy
        if exists_tree_data[2] > 0:
            exists_path = build_dtree(exists_tree, feature, max_depth-1)
        elif exists_tree_data[2] == 0:
            if exists_tree_data[0] == 0: #spam count is 0
                exists_path = node(label='notspam')
            else:
                exists_path = node(label='spam')

        if notexists_tree_data[2] > 0:
            notexists_path = build_dtree(notexists_tree, feature, max_depth-1)
        elif notexists_tree_data[2] == 0:
            if notexists_tree_data[0] == 0: #spam count is 0
                notexists_path = node(label='notspam')
            else:
                notexists_path = node(label='spam')

        return node(word=best_word, frequency=word_freq, exists=exists_path, not_exists=notexists_path)

    # If none of the attribute is able to create non-empty subtrees, max_gain =0 or max depth is reached
    else:
        spam, notspam = get_type_count(file_list)
        if spam > notspam:
            return node(label='spam')
        else:
            return node(label='notspam')

def tree_height (tree):
    """
    Returns the height based on the given node.
    """
    if (tree.exists == None and tree.not_exists == None): return 0
    return max(tree_height(tree.exists), tree_height(tree.not_exists)) + 1


def print_dtree(tree, mode, depth, space=''):
    if tree.label in ('spam', 'notspam'):
        print tree.label
    else:
        if mode == 'binary':
            if depth > 0:
                print tree.word
                print space + 'Exists->',
                print_dtree(tree.exists, mode, depth-1, space + '  ')
                print space + 'Not exists->',
                print_dtree(tree.not_exists, mode, depth-1, space + '  ')
            else:
                print tree.word
        else:
            if depth > 0:
                print tree.word + '>=' + str(tree.frequency)
                print space + 'Yes->',
                print_dtree(tree.exists, mode, depth-1, space + '  ')
                print space + 'No->',
                print_dtree(tree.not_exists, mode, depth-1, space + '  ')
            else:
                print tree.word + '>=' + str(tree.frequency)


def train_dtree(path):
    read_train(path)
    train_files = reduce(getitem, ['spam'], train_file_list) + reduce(getitem, ['notspam'], train_file_list)
    for each_word in words:
        if each_word in doc_freq['notspam'] and each_word in doc_freq['spam']:
            if sum(doc_freq['notspam'][each_word].itervalues()) / float(
                    sum(doc_freq['spam'][each_word].itervalues())) < 0.45 and words[each_word] > 100:
                reduced_words.append(each_word)
        elif each_word not in doc_freq['notspam'] and sum(doc_binary['spam'][each_word].itervalues()) / float(
                len(train_file_list['spam'])) > 0.05:
            reduced_words.append(each_word)

    # print len(reduced_words)
    max_depth = 10
    # feature = ['binary', 'continuous']
    feature = ['continuous', 'binary']
    tree = {}
    for each_feature in feature:
        tree[each_feature] = build_dtree(train_files, each_feature, max_depth)
        print ("Printing the tree for " + each_feature + " feature:")
        print_dtree(tree[each_feature], each_feature, 4)
        visited_words[:] = []
    return tree


# ---------------------------------Testing part--------------------------------

def get_label_bayes(word_list, likelihood, all_words):
    p_spam = math.log(prob_type['spam'])
    p_notspam = math.log(prob_type['notspam'])
    for each_word in word_list:
        each_word = clean_string(each_word)
        # For now ignoring the new words found in test data set
        if len(each_word) > 0 and each_word in all_words:
            p_spam += math.log(likelihood['spam'][each_word])
            p_notspam += math.log(likelihood['notspam'][each_word])

    if p_spam > p_notspam:
        return 'spam'
    else:
        return 'notspam'


def classify_doc(word_list, word_frequency, tree, feature):
    if tree.label != 'NA':
        return tree.label
    else:
        if feature == 'binary':
            if tree.word in word_list:
                sub_tree = tree.exists
            else:
                sub_tree = tree.not_exists
        else:
            if tree.word in word_list and word_frequency[tree.word] >= tree.frequency:
                sub_tree = tree.exists
            else:
                sub_tree = tree.not_exists
    return classify_doc(word_list, word_frequency, sub_tree, feature)


def get_label_dtree(word_list, tree_classifier, feature):
    words_in_doc = []
    frequency = {}
    for each_word in word_list:
        each_word = clean_string(each_word)
        # ignoring the new words found in test data set
        if len(each_word) > 0: #and each_word in words:
            words_in_doc.append(each_word)
            if each_word in frequency:
                frequency[each_word] += 1
            else:
                frequency[each_word] = 1

    return classify_doc(words_in_doc, frequency, tree_classifier, feature)


def get_accuracy(model, likelihood=None, tree_classifier=None, feature=None, all_words=None):
    files = get_list_of_files('./part1/test/')
    correct = 0
    total = 0
    confusion = {"spam":0, "notspam":0, "!spam":0, "!notspam":0}
    path = './part1/test/'
    for each_type in dir_type:
        filelist = files[each_type]
        index = 1
        for each_file in filelist:
            total += 1
            with open(path + each_type + '/''' + each_file) as f:
                read_data = f.read()
                word_list = read_data.split()
                if model == 'bayes':
                    label = get_label_bayes(word_list, likelihood, all_words)
                elif model == 'dtree':
                    label = get_label_dtree(word_list, tree_classifier, feature)
            if label == each_type:
                # print each_file
                confusion[each_type] += 1
                correct += 1
            else: 
                confusion["!"+str(each_type)] += 1
    print_confusion(confusion)
    # print correct, total
    return round(100 * (correct / float(total)), 2)

def print_confusion (confusion):

    confusion_format = "{:>15}" * 4
    print("\n")
    print(confusion_format.format(" ", "Spam", "Not Spam", "Total"))
    print(confusion_format.format("Spam", confusion["spam"], confusion["!spam"], confusion["spam"] + confusion["!spam"]))
    print(confusion_format.format("Not Spam", confusion["!notspam"], confusion["notspam"], confusion["notspam"] + confusion["!notspam"]))
    print(confusion_format.format("Total", confusion["spam"] + confusion["!spam"], confusion["!spam"] + confusion["notspam"], confusion["spam"] + confusion["!spam"] + confusion["notspam"] + confusion["!notspam"]))
    print("\n")


def test_bayes():
    all_words = {}
    # Read all words found in training using any of the likelihood dictionaries
    for each_word in bin_likelihood['spam']:
        if each_word in all_words:
            all_words[each_word] += 1
        else:
            all_words[each_word] = 1
    cont_result = get_accuracy(model='bayes', likelihood=cont_likelihood, all_words=all_words)
    print ("Accuracy for Bayes classifier for continuous feature:" + str(cont_result) + "%")
    bin_result = get_accuracy(model='bayes', likelihood=bin_likelihood, all_words=all_words)
    print ("Accuracy for Bayes classifier for binary feature:" + str(bin_result) + "%")


def test_dtree(trees):
    feature = ['continuous', 'binary']
    for each_feature in feature:
        accuracy = get_accuracy(model='dtree', tree_classifier=trees[each_feature], feature=each_feature)
        print ("Accuracy for decision tree for " + str(each_feature) + " feature:" + str(accuracy))

def display_top_k (dataset, k=10):

    binary_model_spam = sorted(dataset["binary"]["spam"].items(), key=operator.itemgetter(1) , reverse=True)[:10]
    binary_model_notspam = sorted(dataset["binary"]["notspam"].items(), key=operator.itemgetter(1), reverse=True)[:10]
    continuous_model_spam = sorted(dataset["continuous"]["spam"].items(), key=operator.itemgetter(1), reverse=True)[:10]
    continuous_model_notspam = sorted(dataset["continuous"]["notspam"].items(), key=operator.itemgetter(1), reverse=True)[:10]

    print("\nTop 10 Spam Words Most Associated to Spam (Binary):")
    for iterator in range(len(binary_model_spam)):
        print(str(iterator + 1) + ". " + binary_model_spam[iterator][0])

    print("\nTop 10 Spam Words Least Associated to Spam (Binary):")
    for iterator in range(len(binary_model_notspam)):
        print(str(iterator + 1) + ". " + binary_model_notspam[iterator][0])

    print("\nTop 10 Spam Words Most Associated to Spam (Continuous):")
    for iterator in range(len(continuous_model_spam)):
        print(str(iterator + 1) + ". " + continuous_model_spam[iterator][0])

    print("\nTop 10 Spam Words Least Associated to Spam (Continuous):")
    for iterator in range(len(continuous_model_notspam)):
        print(str(iterator + 1) + ". " + continuous_model_notspam[iterator][0])

def main():
    global train_file_list, reduced_words, bin_likelihood, cont_likelihood, prob_type
    
    mode, technique, dataset, model = sys.argv[1:]
    # Check how else this variable can be defined

    start = time.clock()

    train_file_list = get_list_of_files(dataset)

    # For Bayes Classifier:
    if (technique == "bayes"):
        if mode == "train":
            train_bayes(dataset)
            # Build Model
            # bin_likelihood, cont_likelihood, prob_type = train_nb(dataset)
            # Preparing to save all the models for future use
            save_model = {}
            save_model['binary'] = bin_likelihood
            save_model['continuous'] = cont_likelihood
            save_model['prior'] = prob_type

            display_top_k(save_model)

            if (model == None):
                model = './nb_model.p'

            # Saving model for next time re use
            with open(model, 'wb') as handle:
                pickle.dump(save_model, handle, protocol=2)
                handle.close()

            end = time.clock()
            # print(end - start)

        elif mode == "test" and isfile(model):
            # Loading the model
            with open(model, 'rb') as handle:
                complete_model = pickle.load(handle)
                handle.close()
            bin_likelihood = complete_model['binary']
            cont_likelihood = complete_model['continuous']
            prob_type = complete_model['prior']
            test_bayes()

            end = time.clock()
            # print(end - start)

    # For decision tree classifier:
    if (technique == "dt"):
        if mode == "train":
            reduced_words = []
            trees = train_dtree(dataset)
            save_model = {}
            save_model['continuous'] = trees['continuous']
            save_model['binary'] = trees['binary']

            if (model == None):
                model = './nb_model.p'

            # Saving model for next time re use
            with open(model, 'wb') as handle:
                pickle.dump(save_model, handle, protocol=2)
                handle.close()

            end = time.clock()
            # print(end - start)

        elif mode == "test" and isfile(model):
            # Loading the model
            with open(model, 'rb') as handle:
                complete_model = pickle.load(handle)
                handle.close()
            trees = {}
            trees['continuous'] = complete_model['continuous']
            trees['binary'] = complete_model['binary']
            test_dtree(trees)

            end = time.clock()
            # print(end - start)

if __name__ == "__main__":
        main()
