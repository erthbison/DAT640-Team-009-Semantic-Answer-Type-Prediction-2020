import pickle
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from Classes import OntologyType
from ir import EntityCentric
from retrieval_models import BM25_sparse


def select_types(score_type_list: List[Tuple[float, OntologyType]]):
    """Select a list of types out of a provided list of (score, type) tuples
    
    Selects the top scoring type and all types directly higher than it in the hierarchy 
    """
    # Select the type with the highest score, and all its parents
    # Ignore Thing since it is trivial
    selected_type = score_type_list[0][1]
    out_types = []
    while selected_type is not None and selected_type.name != "Thing":
        out_types.append(selected_type.full_name)
        selected_type = selected_type.parent
    return out_types

def load_entity_retrieval(k=100) -> Tuple[EntityCentric, HashingVectorizer]:
    """Loads the data needed for entity retrieval and instantiates objects used"""
    print("Loading data")
    with open("Data/pickle/types-entities.pkl", "rb") as f:
        types, entities = pickle.load(f)
    
    with open("Data/pickle/index.pkl", "rb") as f:
        index = pickle.load(f)

    print("Preparing entity retrieval")

    vectorizer = HashingVectorizer(alternate_sign=False, stop_words="english")

    bm25 = BM25_sparse(index)
    ec = EntityCentric(np.array(types), np.array(entities), bm25, k=k)
    return ec, vectorizer