from typing import Iterable, List, Tuple
from Classes import Entity, OntologyType

import numpy as np
from numpy.typing import ArrayLike

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
            return 1 / type.num_entities
        return 0