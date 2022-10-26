
import json
import pickle
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


from Classes import OntologyType
from ir import EntityCentric
from retrieval_models import BM25_sparse

def select_types(score_type_list: List[Tuple[float, OntologyType]]):
    # Select the type with the highest score, and all its parents
    # Ignore Thing since it is trivial
    selected_type = score_type_list[0][1]
    out_types = []
    while selected_type is not None and selected_type.name != "Thing":
        out_types.append(selected_type.full_name)
        selected_type = selected_type.parent
    return out_types

def load_entity_retrieval(k=100) -> Tuple[EntityCentric, HashingVectorizer]:
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

if __name__ == "__main__":
    

    question_file = "Data/smart-dataset-master/datasets/DBpedia/smarttask_dbpedia_train.json"
    pred_file = "Data/prediction_train.json"

    with open(question_file, "r")  as read_f:
        data = json.load(read_f)

    print("Begin processing questions")
    for k in [75, 100, 125, 150]:
        ec, vectorizer = load_entity_retrieval(k)
        pred_file = f"Data/prediction_train-{k}.json"
        pred_file2 = f"Data/prediction_train2-{k}.json"
        total = len(data)
        out = []
        out2 = []
        for i, q in enumerate(data):
            if (i + 1) % (total // 1000) == 0:
                print(f"\r{round(100*(i/total), 1)}% processed...", end="")
            if q["question"] and q["category"] == "resource":
                query = vectorizer.transform([q["question"]])
                predicted = ec.Score(query)
                pred = {
                    "id": q["id"],
                    "category": q["category"],
                    "type": select_types(predicted)
                }
            else:
                # Just fill in this with the true data just so we get a measure
                pred = {
                    "id": q["id"],
                    "category": q["category"],
                    "type": q["type"]
                }
            out.append(pred)
        with open(pred_file, "w") as write_f:
            json.dump(out, write_f)
        print("Finished processing k:", k)
        
