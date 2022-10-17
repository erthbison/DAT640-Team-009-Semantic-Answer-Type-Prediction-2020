from typing import List, Tuple, Union, Dict
import re
import json
import numpy as np
from numpy import ndarray
import sklearn.feature_extraction.text as sk_text
import sklearn.linear_model as sk_linear

#Data manipulation function

def open_data(filename): #opens the datafile and creates a list of json objects (strings)
    f = open(filename,'r')
    lines = f.readlines()
    add = False
    list_of_json = [""]
    for line in lines:
        if "}" in line and len(line) == 5:
            list_of_json[-1] += "}"
            add = False
            list_of_json.append("")
        if "{" in line and len(line) == 4:
            add = True
        if add:
            list_of_json[-1] += line
    f.close()
    list_of_json[-1] = list_of_json[-1][:-1]
    return list_of_json
        
def parse(json_data): #returns a data object with attributes "id","question"...
    return json.loads(json_data)

def target_type(json_object): #this function encapsulates the types to facilitate the ML process
    #It returns the target type of the json_object
    #boolean, string, date, number or resource
    if json_object["category"] == "resource":
        return json_object["category"]
    else:
        return json_object["type"][0]


def preprocess(doc: str) -> List[str]: #Preprocessing without stopwords
    return [
        term
        for term in re.sub(r"[^\w]|_", " ", doc).lower().split()
    ]
def question_target(filename): #Takes the dataset and generates a list of [question,type]
    questions,targets = [],[]
    for elt in open_data(filename):
        parsed = parse(elt)
        question,target = parsed["question"],target_type(parsed)
        questions.append(" ".join(preprocess(str(question))))
        targets.append(target)
    return questions,targets

def extract_features(
    train_dataset: List[str], test_dataset: List[str]
) -> Union[Tuple[ndarray, ndarray], Tuple[List[float], List[float]]]:
    """Extracts feature vectors from a preprocessed train and test datasets.

    Args:
        train_dataset: List of strings, each consisting of the preprocessed
            email content.
        test_dataset: List of strings, each consisting of the preprocessed
            email content.

    Returns:
        A tuple of of two lists. The lists contain extracted features for 
          training and testing dataset respectively.
    """
    vectorizer = sk_text.TfidfVectorizer()
    X =  vectorizer.fit_transform(train_dataset+test_dataset) #We have to provide the union of both vocabularies in order to train
    X_train,X_test = X[:len(train_dataset),],X[len(train_dataset):,]
    return X_train,X_test