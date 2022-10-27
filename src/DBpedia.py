"""Contains parsers for DBpedia files."""

from Classes import OntologyType, Entity

import pickle
from typing import Dict
from sklearn.feature_extraction.text import HashingVectorizer


def get_types(filename: str) -> Dict[str, OntologyType]:
    """Retrieve all DBpedia types from the provided files.
    
    Return a Dict storing type name and OntologyType objects for the types.
    The num_entities value is not inserted as no entities have been read.
    """
    types = {"Thing": OntologyType("Thing", "rdf")}
    types_parents = {"Thing": ""}
    with open(filename, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip("\n. ")
            try:
                s, p, o = line.split(" ", 2)
                if "http://www.w3.org/2002/07/owl#Class" in o:
                    name = strip_dbpedia_url(s, "ontology")
                    types[name] = OntologyType(name, "dbo")
                if "http://www.w3.org/2000/01/rdf-schema#subClassOf" in p:
                    # Get the types name
                    name = strip_dbpedia_url(s, "ontology")
                    # Get the parents name
                    parent = strip_ontology_url(o)
                    # Store the link between the type and its parent in a dictionary
                    types_parents[name] = parent
            except ValueError:
                continue
    # Assign the correct type as the parent for all types
    for ontology_type in types.values():
        parent_name = types_parents[ontology_type.name]
        ontology_type.parent = types.get(parent_name, None)
    return types

def get_entities(filename, types: Dict[str, OntologyType]):
    """Get a list of all entities from a instance_types_en.ttl file
    
    Also contains information about the type of each entity and updates num_entities values of all provided types 
    """
    # Store all entities in a dictionary
    entities: Dict[str, Entity] = {}
    # Get all resources/entities and link them to a type
    with open(filename, "r", encoding="utf-8") as f:
        f.__next__() # skips header file
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("Entity-Types: Reading line:", i)
            line = line.strip("\n. ")
            try:
                s, p, o = line.split(" ", 2)
                if "http://dbpedia.org/ontology/" not in o and "http://www.w3.org/2002/07/owl#" not in o:
                    continue
                name = strip_dbpedia_url(s, "resource")
                type = strip_ontology_url(o)
                ontology_type = types[type]
                entities[name] = Entity(name, ontology_type)
                # increment number of entities in type
                ontology_type.num_entities += 1
            except ValueError:
                continue
    print("Completed reading entity types")
    return entities

def extract_literal(entity_repr: Dict[str, str], filename: str):
    """Extracts literal value from file and adds it to the the entity representation of its respective entity"""
     # Get abstract data for all entities
    with open(filename, "r", encoding="utf-8") as f:
        f.__next__()
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print(f"\rExtracting literal - {f.name}:", i, end="")
            line = line.strip("\n. ")
            try:
                s, p, o = line.split(" ", 2)
                name = strip_dbpedia_url(s, "resource")
                e = entity_repr.get(name, None)
                if e is None:
                    continue
                e += f" {o}"
                entity_repr[name] = e
            except ValueError:
                continue
    print("Completed reading entity abstracts")

def extract_uri(entity_repr: Dict[str, str], filename: str):
    """Extracts uri value from file, strips the uri and adds the remainder to the entity representation of its respective entity"""
    with open(filename, "r", encoding="utf-8") as f:
        f.__next__()
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print(f"\rExtracting uri - {f.name}:", i, end="")
            try:
                s, p, o = line.split(" ", 2)
                name = strip_dbpedia_url(s, "resource")
                redirect = strip_dbpedia_url(o, "resource")
                e = entity_repr.get(name, None)
                if e is None:
                    continue
                e += f" {redirect}"
                entity_repr[name] = e
            except ValueError:
                continue
    print("Completed reading entity redirects")

def strip_ontology_url(url) -> str:
    """Strips the uri of an ontology element"""
    url = url.strip("<>")
    if "http://dbpedia.org/ontology/" in url:
        return url.replace("http://dbpedia.org/ontology/", "")
    elif "http://www.w3.org/2002/07/owl#" in url:
        return url.replace("http://www.w3.org/2002/07/owl#", "")

def strip_dbpedia_url(url: str, type:str) -> str:
    """Strips the uri of an element"""
    url= url.strip("<>")
    url = url.replace(f"http://dbpedia.org/{type}/", "")
    return url.strip("/")

if __name__ == "__main__":
    nt = get_types("Data/dbpedia/dbpedia_2016-10.nt")
    entities = get_entities("Data/dbpedia/instance_types_en.ttl", nt)

    # Creates a basic entity representation for all entities
    entity_repr = {e.name: f"{e.name} {e.ontology_type.name}" for e in entities.values()}


    # Add to entity representation
    extract_literal(entity_repr, "Data/dbpedia/long_abstracts_en.ttl")
    extract_uri(entity_repr, "Data/dbpedia/redirects_en.ttl")
    extract_uri(entity_repr, "Data/dbpedia/disambiguations_en.ttl")
    extract_uri(entity_repr, "Data/dbpedia/article_categories_en.ttl")

    # index entity representation using Hashing vectorizer
    print("Creating index")
    vectorizer = HashingVectorizer(alternate_sign=False, stop_words="english")
    index = vectorizer.transform([e for  e in entity_repr.values()])

    print("Pickling index")
    with open("Data/pickle/index.pkl", "wb") as f:
        pickle.dump(index, f)

    print("Pickling entity and type lists")
    with open("Data/pickle/types-entities.pkl", "wb") as f:
        pickle.dump((list(nt.values()), list(entities.values())), f)
    print("Finished pickling data")