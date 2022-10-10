from typing import List, Tuple, Union, Dict
import re
import numpy as np
from numpy import ndarray
import sklearn.feature_extraction.text as sk_text
import sklearn.linear_model as sk_linear

def preprocess(doc: str) -> List[str]: #Preprocessing without stopwords
    return [
        term
        for term in re.sub(r"[^\w]|_", " ", doc).lower().split()
    ]


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