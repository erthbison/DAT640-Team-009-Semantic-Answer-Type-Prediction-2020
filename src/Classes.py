from dataclasses import dataclass, field
from typing import Tuple, Union


@dataclass(order=True)
class OntologyType():
    """A class representing a type from an ontology"""
    name: str
    prefix: str
    parent: Tuple["OntologyType", None] = field(default=None, repr=False)
    num_entities: int = 0

    @property
    def full_name(self):
        return f"{self.prefix}:{self.name}"

@dataclass(order=True)
class Entity():
    """A class representing an Entity.
    
    Does not contain the entity representation"""
    name: str
    ontology_type: Union["OntologyType", None] = field(default=None, repr=False)
