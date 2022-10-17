from typing import Tuple, Iterable, Union, Dict
from dataclasses import dataclass, field

@dataclass(order=True)
class OntologyType():
    name: str
    prefix: str
    order: int = -1
    parent: Tuple["OntologyType", None] = field(default=None, repr=False)
    entities: Iterable["Entity"] = field(default_factory=list, repr=False)

    @property
    def full_name(self):
        return f"{self.prefix}:{self.name}"

@dataclass(order=True)
class Entity():
    name: str
    ontology_type: Union["OntologyType", None] = None

    abstract: str = field(default="", repr=False)
    alternative_names: str = field(default="", repr=False)

    def get_document(self):
        return f"{self.name} - {self.ontology_type.name} - {self.abstract} - {self.alternative_names}\n"