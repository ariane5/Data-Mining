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
from sklearn.svm import SVC
from nltk.classify import DecisionTreeClassifier
from senti_classifier import senti_classifier
from nltk.stem import PorterStemmer, WordNetLemmatizer

training = []
trainingPositive = []
trainingNegative = []
port = PorterStemmer()

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
                                word1=port.stem(word)
				new_word = processText(word1)
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
                                word2=port.stem(word1)
				new_word = processText(word2)
	    			if new_word is not None:
				         my_list1.append(new_word)
                                else:
                                         continue						
		trainingNegative.append((my_list1,'negative'))
        #print trainingNegative

list_ = []
for (words, sentiment) in trainingPositive + trainingNegative:
           if len(words) >= 3:
              list_.append((words, sentiment))
          
random.shuffle(list_)

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

#classifier1=SklearnClassifier(SVC(), sparse=False).train(training_set)


print(nl.classify.accuracy(classifier, test_set))
#print(nl.classify.accuracy(classifier1, test_set))

f = open('pos/cv000_29590.txt', 'r')#cv001_19502.txt
review_movie=f.read()
print classifier.classify(get_extract_features(review_movie.split()))
print(nl.classify.accuracy(classifier, test_set))





