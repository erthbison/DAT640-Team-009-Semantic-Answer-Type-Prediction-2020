from typing import Iterable, List
from numpy.typing import ArrayLike
from preprocess_data import InvertedIndex, FieldedInvertedIndex
import math
import numpy as np

from scipy.sparse import csc_matrix


# Some implementations of retrieval functions. These are not very performant and is currently not used. 

class BM25():
    def __init__(self, index: InvertedIndex, entities,k1:float=1, b: float=1) -> None:
        self.k1 = k1
        self.b = b
        self.index = index

        self.entities = entities

        entity_len = {}
        total = 0
        for term, entity_dict in index.items():
            for entity_name, freq in entity_dict.items():
                total += freq
                entity_len[entity_name] = entity_len.get(entity_name, 0) + freq

        self.__avg_len = total / len(entity_len)
        
        self.__length_array =  np.array([entity_len.get(entity.name, 0) for entity in entities])

    def Score(self, query) -> ArrayLike:
        return np.sum(self.score_term(term) for term in query)

    def score_term(self, term) -> ArrayLike:
        c_array = np.array([self.index.get_term_frequency(term, entity.name) for entity in self.entities])
        return c_array * (1 + self.k1) / (c_array + self.k1 * (1 - self.b + self.b * self.__length_array / self.__avg_len)) * self.__idf(term)

    def __idf(self, term):
        return math.log(len(self.__length_array) / self.index.get_num_postings(term))

class BM25f():
    def __init__(self, index: FieldedInvertedIndex, fields: Iterable[str], field_weights: Iterable[float], k1: float = 1, b: float = 1) -> None:
        assert sum(field_weights) == 1
        self.k1 = k1
        self.b = b
        self.fields = fields
        self.field_weights = field_weights

        self.index = index

        entity_len = {}
        self.__avg_field_len = {}
        for field in fields:
            total = 0
            for term, entity_dict in index[field].items():
                for entity_name, freq in entity_dict.items():
                    temp = entity_len.get(field, {})
                    temp[entity_name] = temp.get(entity_name, 0) + freq
                    entity_len[field] = temp
                    total += freq
            self.__avg_field_len[field] = total / len(entity_len[field])
        self.__entity_len = entity_len

    def Score(self, query, entity) -> List[float]:
        return sum(self.score_term(term, entity) for term in query)

    def score_term(self, term, entity) -> float:
        idf = self.__idf(term)
        
        def field_pseudo_count(args):
            i, field = args
            bi = 1 - self.b + self.b * self.__entity_len[field].get(entity.name, 0) / self.__avg_field_len[field]
            return self.field_weights[i] * self.index.get_term_frequency(field, entity.name, term) / bi

        c = sum(map(field_pseudo_count, enumerate(self.fields)))
        return c / (self.k1 + c) * idf
        
    def __idf(self, term):
        field = "name"
        return math.log(len(self.__entity_len[field]) / self.index.get_num_postings(field, term))


class BM25_sklearn:
    def __init__(self, index, k1=1.2, b=0.75):
        # document-term matrix. row: document, column: terms
        self.index: csc_matrix = csc_matrix(index)
        self.k1: float = k1
        self.b: float = b

    def get_scores(self, query: csc_matrix):
        num_docs, num_terms = self.index.shape
        total = np.zeros(num_docs)
        _, terms = query.nonzero()
        # for term in terms:
            # # For all terms in query
            # ctd = self.index.getcol(term)
            # d = self.index.sum(-1)
            # avgd = d.sum() / num_docs
            # if avgd == 0:
            #     print("HERER")
            # idf = math.log(num_docs / ctd.count_nonzero())
            # denominator = (ctd + self.k1 * (1 - self.b + self.b * d / avgd))
            # if denominator.any(0):
            #     print("HERE2")
            # temp = ctd * (1 + self.k1) / denominator * idf
            # total += temp 
        ctd = self.index[:, terms]

        # Get the number of elements in each document/entity(row)
        d = self.index.sum(-1)
        avgd = d.sum() / num_docs
        # Broadcast the idf to the same shape as cdf. The idf is the same for all entities 
        idf = np.log(num_docs / np.sum(ctd != 0, 0))
        # idf = np.broadcast_to(test, ctd.shape)
        test = self.k1 * (1 - self.b + self.b * d / avgd)
        denominator = (ctd + test)
        temp = ctd.multiply((1 + self.k1))
        w = temp.multiply(idf)
        t = w / denominator
        total = t.sum(-1)
        return np.array(total).reshape(-1)
