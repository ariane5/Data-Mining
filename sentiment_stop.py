#!/usr/bin/python
import glob
import numpy as np
import re
import nltk as nl
import random
from collections import defaultdict
from nltk.classify import apply_features
from nltk.probability import FreqDist, DictionaryProbDist, ELEProbDist, sum_logs
from nltk.classify import NaiveBayesClassifier
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import SklearnClassifier
#classif = SklearnClassifier(LinearSVC())
#nltk.classify.svm.SvmClassifier(*args, **kwargs)
from sklearn.svm import SVC
from nltk.classify import DecisionTreeClassifier
from senti_classifier import senti_classifier

training = []
trainingPositive = []
trainingNegative = []


def processText(word):
	if word not in stopWords:
		if re.match("^[A-Za-z]*$", word):
			return word

with open('stopwords.txt') as text1:
    stopWords = [word.strip() for word in text1]
    for filename in glob.glob('pos/*.txt'):
        with open(filename) as text:
		my_list = []
           	for line in text:
			for word in line.split():
				new_word = processText(word)
				if new_word is not None:
					my_list.append(new_word)
				else: 
					continue
		trainingPositive.append((my_list,'positive'))
	#print trainingPositive[0]

    for filename1 in glob.glob('neg/*.txt'):
        with open(filename1) as text1:
		my_list1 = []
           	for line1 in text1:
                	for word1 in line1.split(): # pos tokenization
                                new_word1 = processText(word1)
	    			if new_word1 is not None:
				         my_list1.append(new_word1)
                                else:
                                         continue						
		trainingNegative.append((my_list1,'negative'))
        #print trainingNegative

list_ = []
for (words, sentiment) in trainingPositive + trainingNegative:
           if len(words) >= 3:
              list_.append((words, sentiment))
          
random.shuffle(list_)
#print list_[:20]

#The list of word features need to be extracted from the review. It is a list with every distinct words ordered by frequency of appearance. We use the following function to get the list plus the two helper functions.

def get_words_in_list_(document):
    all_words = []
    for (words, sentiment) in list_:
      all_words.extend(words)
    return all_words

all_words=get_words_in_list_(list_)


#print all_words
def get_word_features(wordlist):
    wordlist = nl.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

word_features = get_word_features(get_words_in_list_(list_))#37643
#print word_features[:100]

#To create a classifier, we need to decide what features are relevant. To do that, we first need a feature extractor. The one we are going to use returns a dictionary indicating what words are contained in the input passed. Here, the input is the tweet. We use the word features list defined above along with the input to create the dictionary.



def get_extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features[:10000]:#we reduce the number of word_feature to making it work fast
        features['contains(%s)' % word] = (word in document_words)
    return features

extract_features= get_extract_features(word_features)

#print extract_features




training_set = nl.classify.apply_features(get_extract_features, list_[:1500])
test_set = nl.classify.apply_features(get_extract_features, list_[1500:])
#print training_set

classifier = nl.NaiveBayesClassifier.train(training_set)
#print classifier.show_most_informative_features(100)

#classifier1=SklearnClassifier(SVC(), sparse=False).train(training_set)#
#print len(list_[1500:])



#print(nl.classify.accuracy(classifier, test_set))








#classify

f = open('pos/cv000_29590.txt', 'r')#cv001_19502.txt
review_movie=f.read()
#print classifier.classify(get_extract_features(review_movie.split()))
#pos_score, neg_score = senti_classifier.polarity_scores(review_movie)
#print pos_score, neg_score

#for filename in glob.glob('pos/*.txt'):
 #       with open(filename) as f:
#             review_movie=f.read()
#             if classifier.classify(get_extract_features(review_movie.split())):
#                 counter=counter+1
#             else:  
#                 counter1 =counter1+1

#print counter 
#print counter1
#error=counter1/1000
#print error 
    ## put that in th e other folder 
#classifier1=SklearnClassifier(SVC(), sparse=False).train(training_set)

print classifier.classify(get_extract_features(review_movie.split()))
print(nl.classify.accuracy(classifier, test_set))

#classifier2=nl.classify.DecisionTreeClassifier.train(training_set, entropy_cutoff=0,support_cutoff=0)
#print(nl.classify.accuracy(classifier2, test_set))
#print classifier2.classify(get_extract_features(review_movie.split()))


