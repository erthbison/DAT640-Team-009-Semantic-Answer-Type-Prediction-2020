from typing import Tuple, Iterable, Union
from dataclasses import dataclass, field

@dataclass(order=True)
class OntologyType():
    name: str
    prefix: str
    order: int = -1
    parent: Tuple["OntologyType", None] = field(default=None, repr=False)
    num_entities: int = 0

    @property
    def full_name(self):
        return f"{self.prefix}:{self.name}"

@dataclass(order=True)
class Entity():
    name: str
    ontology_type: Union["OntologyType", None] = None