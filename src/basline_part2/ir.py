from typing import Iterable, List, Tuple
from Classes import Entity, OntologyType
from bisect import insort_left
import pickle
import json
from preprocess_data import preprocess, InvertedIndex, FieldedInvertedIndex

from rank_bm25 import BM25Okapi
from retrieval_models import BM25, BM25f

import numpy as np
from numpy.typing import ArrayLike

import cProfile
import pstats

class TypeCentric():
    def __init__(self, types, entities) -> None:
        self.types: List[OntologyType] = types
        self.entities: List[Entity] = entities
        self.k1 = 1.2
        self.b = 0.75
        self.avg_doc_length = sum(len(e.abstract) for e in self.entities) / len(self.entities)

    def Score_all_types(self, query: List[str]) -> List[OntologyType]:
        ranked_types = []
        for i, type in enumerate(self.types):
            if i == 10:
                break
                print("Processing type number", i, "current top 10", ranked_types[-10:])
            score = self.score_query(query, type)
            print(score)
            insort_left(ranked_types, (score, type))
        return ranked_types

    def score_query(self, query: List[str], type: OntologyType):
        total = 0
        for term in query:
            count = self.pseudo_count(term, type)
            total += (count * (1 + self.k1) / 
                (count + self.k1 * (1- self.b + (self.b*sum(len(e.abstract) for e in type.entities))) / self.avg_doc_length)
            )
        return total

    def pseudo_count(self, term:str, type:OntologyType):
        total = 0
        for entity in self.entities:
            total += entity.counter.get(term, 0) * self.weight(entity, type)
        return total

    def weight(self, entity: Entity, type: OntologyType):
        if entity.ontology_type == type:
            return 1 / len(type.entities)
        return 0


class EntityCentric():
    def __init__(self, types: ArrayLike, entities: ArrayLike, retrieval_model, k:int = 10, num_types: int = 10) -> None:
        self.types = types
        self.entities = entities
        self.retrieval_model = retrieval_model
        self.k = k
        self.num_types = num_types
        
    def Score(self, query: Iterable[str]) -> List[Tuple[float, OntologyType]]:
        # Score all entities
        entity_score = self.retrieval_model.get_scores(query)

        # Find the top k entities and their corresponding scores
        top_indices = np.argpartition(entity_score, -self.k)[-self.k:]
        top_entities = self.entities[top_indices]
        top_entity_scores = entity_score[top_indices]
        
        # Calculate score for all types
        # TODO: Consider only calculating scores for types with at least one entity. Would probably save significant amount of time
        type_scores = np.fromiter(map(lambda x: self.score_type(x, top_entities, top_entity_scores), self.types), np.float64)

        # Find the types with the num_types highest scores and retrieve the score and types
        top_indices = np.argpartition(type_scores, -self.num_types)[-self.num_types:]
        top_types = self.types[top_indices]
        top_type_scores = type_scores[top_indices]

        # Return the type scores and types in sorted order
        return sorted([(score, type) for score, type in zip(top_type_scores, top_types)], reverse=True)

    def score_type(self, type, top_entities, top_entity_score):
        weights = np.fromiter(map(self.weight, top_entities, [type] * self.k), np.float64)
        return np.sum(np.multiply(top_entity_score,  weights))

    def weight(self, entity: Entity, type: OntologyType):
        if entity.ontology_type == type:
            return 1 / len(type.entities)
        return 0

if __name__ == "__main__":
    print("Loading data")
    with open("types-entities.pkl", "rb") as f:
        types, entities = pickle.load(f)

    print("Loading question")
    with open("Data/smart-dataset-master/datasets/DBpedia/smarttask_dbpedia_test.json", "r") as f:
        data = json.load(f)
        for q in data:
            question = q["question"]
            type = q["type"]
            if q["category"] == "resource": break
    print("Pre-processing question")
    query = preprocess(question)



    # Score with BM25
    # with open("Data/pickle/entity-index.pkl", "rb") as f:
    #     index = pickle.load(f)

    with open("Data/pickle/entity-term-list.pkl", "rb") as f:
        index = pickle.load(f)


    print("Preparing entity retrieval")
    # bm25 = BM25(index, entities, b=0.75, k1=1.20)
    # bm25 = BM25Okapi(index)
    ec = EntityCentric(np.array(types), np.array(entities), bm25, k=1000)
    with cProfile.Profile() as pr:
        print("Scoring query")
        pr.enable()
        results = ec.Score(query)
        pr.disable()

        with open("profiler_bm25.txt", "w") as f:
            sortby = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=f).sort_stats(sortby)
            ps.print_stats()

    print(f"Question: {question}")
    print(f"Type: {type}")
    for type_score, type in results:
        if type_score > 0:
            print(f"{type_score}: {type}")

    # # Score with BM25F
    # with open("Data/pickle/entity-fielded-index.pkl", "rb") as f:
    #     index = pickle.load(f)
    # print("Preparing entity retrieval")
    # bm25 = BM25f(index, fields=["name", "alternative_names"], field_weights=[0.75, 0.25], b=0.75, k1=1.20)
    # ec = EntityCentric(types, entities, bm25, k=100)
    # print("Scoring query")
    # results = ec.Score(query)

    # print(f"Question: {question}")
    # print(f"Type: {type}")
    # for type_score, type in results:
    #     if type_score > 0:
    #         print(f"{type_score}: {type}")