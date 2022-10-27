from typing import Iterable, List, Tuple

import numpy as np
from numpy.typing import ArrayLike

from Classes import Entity, OntologyType


class EntityCentric():
    def __init__(self, types: np.ndarray[OntologyType], entities: np.ndarray[Entity], retrieval_model, k:int = 10, num_types: int = 10) -> None:
        """Score a list of types according to an entity centric model.
        
        types: A numpy array of the types to be scored
        entities: A numpy array of all entities belonging to the types
        retrieval_model: A retrieval model used to score the entities. The retrieval model should implement the method get_scores(query) and 
            should be pre-instantiated with an index. The score function should provide an numpy array of scores, where score i should match the i-th entity of entities
        k: How many entities to consider when calculating the score for the types
        num_types: How many (score, type) tuples should be returned. 
        """
        self.types = types
        self.entities = entities
        self.retrieval_model = retrieval_model
        self.k = k
        self.num_types = num_types
        
    def Score(self, query: Iterable[str]) -> List[Tuple[float, OntologyType]]:
        """Score all types for the provided query and return a sorted list of (score, type) tuples for the top num_types types
        
        Uses an entity-centric model where a score is first calculated for all entities using the provided retrieval model. 
        Then selects the top k scoring entities and uses these entities to calculate a score for each type.
        The score for each type(y) is the sum of the product between each entity(e) and a weight(w) for that entity: score(y, q) = sum(score(e, q) * w(e, y)).
        The weight is set to uniformly weight all entities of a type. Entities not belonging to the type has a weight of 0. 
        """
        # Score all entities
        entity_score = self.retrieval_model.get_scores(query)

        # Find the top k entities and their corresponding scores
        top_indices = np.argpartition(entity_score, -self.k)[-self.k:]
        top_entities = self.entities[top_indices]
        top_entity_scores = entity_score[top_indices]
        
        # Calculate score for all types
        type_scores = np.fromiter(map(lambda x: self.score_type(x, top_entities, top_entity_scores), self.types), np.float64)

        # Find the types with the num_types highest scores and retrieve the score and types
        top_indices = np.argpartition(type_scores, -self.num_types)[-self.num_types:]
        top_types = self.types[top_indices]
        top_type_scores = type_scores[top_indices]

        # Return the type scores and types in sorted order
        return sorted([(score, type) for score, type in zip(top_type_scores, top_types)], reverse=True)

    def score_type(self, type, top_entities, top_entity_score):
        """Score the provided type with the provided entities and entity scores.
        
        The score for each type(y) is the sum of the product between each entity(e) and a weight(w) for that entity: score(y, q) = sum(score(e, q) * w(e, y)).
        """
        weights = np.fromiter(map(self.weight, top_entities, [type] * self.k), np.float64)
        return np.sum(np.multiply(top_entity_score,  weights))

    def weight(self, entity: Entity, type: OntologyType):
        """Calculate the weight of the provided type and entity.
        
        The weight is set to uniformly weight all entities of a type. Entities not belonging to the type has a weight of 0. 
        """
        if entity.ontology_type == type:
            return 1 / type.num_entities
        return 0
