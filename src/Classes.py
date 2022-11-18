from dataclasses import dataclass, field
from typing import Tuple, Union


@dataclass(order=True)
class OntologyType():
	"""
	OntologyType class

	Attributes
	----------
	name : str
		Name of the ontology type
	prefix : str
		Prefix of the ontology type
	parent : str
		Parent of the ontology type
	num_entities : int
		Number of entities of the ontology type

	Methods
	-------
	full_name()
		Returns the full name of the ontology type using it's suffix and name
	"""
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
