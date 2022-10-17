from Classes import OntologyType, Entity

import pickle

def get_types(filename):
    types = {"Thing": OntologyType("Thing", "rdf", 0, None)}
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
    for ontology_type in types.values():
        parent_name = types_parents[ontology_type.name]
        ontology_type.parent = types.get(parent_name, None)
    for ontology_type in types.values():
        parent_type = ontology_type.parent
        rank = 0
        while parent_type is not None:
            rank += 1
            parent_type = parent_type.parent
        ontology_type.order = rank
    return types

def get_entities(instanceof_filename, abstracts_filename, types):
    entities = {}
    # Get all resources/entities and link them to a type
    with open(instanceof_filename, "r", encoding="utf-8") as f:
        f.__next__()
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
                e = Entity(name, ontology_type)
                entities[name] = e
                ontology_type.entities.append(e)
            except ValueError:
                continue
    print("Completed reading entity types")
    # Get abstract data for all entities
    with open(abstracts_filename, "r", encoding="utf-8") as f:
        f.__next__()
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("Entity-Abstracts: Reading line:", i)
            line = line.strip("\n. ")
            try:
                s, p, o = line.split(" ", 2)
                name = strip_dbpedia_url(s, "resource")
                e = entities.get(name, None)
                if e is None:
                    continue
                e.abstract = o
            except ValueError:
                continue
    print("Completed reading entity abstracts")
    return entities


def get_redirects(entities, filename):
    with open(filename, "r", encoding="utf-8") as f:
        f.__next__()
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("Entity-Redirects: Reading line:", i)
            try:
                s, p, o = line.split(" ", 2)
                name = strip_dbpedia_url(s, "resource")
                redirect = strip_dbpedia_url(o, "resource")
                e = entities.get(name, None)
                if e is None:
                    continue
                e.alternative_names += f" {redirect}"
            except ValueError:
                continue

def strip_ontology_url(url) -> str:
    url = url.strip("<>")
    if "http://dbpedia.org/ontology/" in url:
        return url.replace("http://dbpedia.org/ontology/", "")
    elif "http://www.w3.org/2002/07/owl#" in url:
        return url.replace("http://www.w3.org/2002/07/owl#", "")

def strip_dbpedia_url(url: str, type:str) -> str:
    url= url.strip("<>")
    url = url.replace(f"http://dbpedia.org/{type}/", "")
    return url.strip("/")

if __name__ == "__main__":
    nt = get_types("Data/dbpedia/dbpedia_2016-10.nt")
    entities = get_entities("Data/dbpedia/instance_types_en.ttl", "Data/dbpedia/short_abstracts_en.ttl", nt)

    get_redirects(entities, "Data/dbpedia/redirects_en.ttl")

    with open("types-entities.pkl", "wb") as f:
        pickle.dump((list(nt.values()), list(entities.values())), f)
    print("Finished pickling data")