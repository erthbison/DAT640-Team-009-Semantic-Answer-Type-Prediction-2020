from typing import Iterable, List
from numpy.typing import ArrayLike
import math
import numpy as np

from scipy.sparse import csc_matrix

class BM25_sparse:
    def __init__(self, index, k1=1.2, b=0.75):
        # document-term matrix. row: document, column: terms
        self.index: csc_matrix = csc_matrix(index)
        self.k1: float = k1
        self.b: float = b

        num_docs, num_terms = self.index.shape

        d = self.index.sum(-1)
        avgd = d.sum() / num_docs
        # Precalculate some values
        self.__denom = self.k1 * (1 - self.b + self.b * d / avgd)

        # Get number of nonzero elements of each column, i.e. number of documents with that term in it
        self.__term_in_docs = index.getnnz(0)
        

    def get_scores(self, query: csc_matrix) -> np.ndarray:
        query = csc_matrix(query)
        num_docs, num_terms = self.index.shape
        _, terms = query.nonzero()
        
        # Get the term counts for the terms in the query
        ctd = self.index[:, terms]

        # Calculate idf for terms in the query
        idf = np.log(num_docs / self.__term_in_docs[terms])

        # Calculate the score for each term for each document/entity
        score_per_term = ctd.multiply(1 + self.k1).multiply(idf) / (ctd + self.__denom)
        # Sum the score of each term in each document into a total score for each entity
        total_score = score_per_term.sum(-1)
        # Reshape into a numpy array
        return np.array(total_score).reshape(-1)
