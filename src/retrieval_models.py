import numpy as np

from scipy.sparse import csc_matrix

class BM25_sparse:
    def __init__(self, index: csc_matrix, k1=1.2, b=0.75):
        """Implementation of the BM25 retrieval model using sparse document-term matrix representation for the index.
        
        The class implements the interface for a retrieval model described in EntityCentric.
        """
        # document-term matrix. row: document, column: terms
        self.index: csc_matrix = csc_matrix(index)
        self.k1: float = k1
        self.b: float = b

        self.__num_docs, self.__num_terms = self.index.shape

        d = self.index.sum(-1)
        avgd = d.sum() / self.__num_docs
        # Precalculate some values
        self.__denom = self.k1 * (1 - self.b + self.b * d / avgd)

    def get_scores(self, query: csc_matrix) -> np.ndarray[float]:
        """Score all documents for the provided query"""
        query = csc_matrix(query)
        _, terms = query.nonzero()
        
        # Get the term counts for the terms in the query
        ctd = self.index[:, terms]

        # Calculate idf for terms in the query
        idf = np.log(self.__num_docs / ctd.getnnz(0))

        # Calculate the score for each term for each document/entity
        score_per_term = ctd.multiply(1 + self.k1).multiply(idf) / (ctd + self.__denom)
        # Sum the score of each term in each document into a total score for each entity
        total_score = score_per_term.sum(-1)
        # Reshape into a numpy array
        return np.array(total_score).reshape(-1)
