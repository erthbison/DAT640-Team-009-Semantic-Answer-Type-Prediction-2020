from concurrent.futures import ProcessPoolExecutor
import json
import pickle
from collections import defaultdict
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from Classes import OntologyType
from ir import EntityCentric
from retrieval_models import BM25_sparse

# For progress barr
import tqdm

def select_types(score_type_list: List[Tuple[float, OntologyType]]):
    # Select the type with the highest score, and all its parents
    # Ignore Thing since it is trivial
    selected_type = score_type_list[0][1]
    out_types = []
    while selected_type is not None and selected_type.name != "Thing":
        out_types.append(selected_type.full_name)
        selected_type = selected_type.parent
    return out_types

VECTORIZER = None
EC = None

def init(k):
    global VECTORIZER
    global EC
    with open("Data/pickle/types-entities.pkl", "rb") as f:
        types, entities = pickle.load(f)
    
    with open("Data/pickle/index.pkl", "rb") as f:
        index = pickle.load(f)

    VECTORIZER = HashingVectorizer(alternate_sign=False, stop_words="english")

    bm25 = BM25_sparse(index)
    EC = EntityCentric(np.array(types), np.array(entities), bm25, k=k)

def handle_question(q):
    global VECTORIZER
    global EC
    if q["question"] and q["category"] == "resource":
        query = VECTORIZER.transform([q["question"]])
        predicted = EC.Score(query)
        return {
            "id": q["id"],
            "category": q["category"],
            "type": select_types(predicted)
        }
    else:
        # Just fill in this with the true data just so we get a measure
        return {
            "id": q["id"],
            "category": q["category"],
            "type": q["type"]
        }
    

if __name__ == "__main__":
    print(__name__)

    question_file = "Data/smart-dataset-master/datasets/DBpedia/smarttask_dbpedia_train.json"
    
    with open(question_file, "r")  as read_f:
        data = json.load(read_f)

    print("Begin processing questions")
    for k in [5, 10, 25, 50]:
        pred_file = f"Data/prediction_train_future-{k}.json"
        out = []
        with ProcessPoolExecutor(4, initializer=init, initargs=[k]) as p:
            for pred in tqdm.tqdm(p.map(handle_question, data, chunksize=10), total=len(data)):
            # for pred in p.map(handle_question, data):
                out.append(pred)

        with open(pred_file, "w") as write_f:
            json.dump(out, write_f)