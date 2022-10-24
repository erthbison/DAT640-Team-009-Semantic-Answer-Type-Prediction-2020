
import json
import pickle
from collections import defaultdict
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from Classes import OntologyType
from ir import EntityCentric
from retrieval_models import BM25_sklearn

# For progress bar. TODO: REMOVE
import tqdm

def select_types(score_type_list: Iterable[Tuple[float, OntologyType]]) -> List[str]:
    # from the list of types return a list of types on the same branch with the highest total score
    type_dict = {}
    score_dict = defaultdict(lambda: 0, {t.name: s for s, t in score_type_list})
    aggregated_score = defaultdict(lambda: 0)
    for _, type in score_type_list:
        name = type.name
        type_dict[name] = type
        while type is not None:
            aggregated_score[name] += score_dict[name]
            type = type.parent
    type_name, _ = max(aggregated_score.items(), key=lambda x: x[1])
    selected_type = type_dict[type_name]
    out_types = []
    while selected_type is not None and selected_type.name != "Thing":
        if selected_type.name in aggregated_score:
            out_types.append(selected_type.full_name)
        selected_type = selected_type.parent
    return out_types

if __name__ == "__main__":
    print("Loading data")
    with open("Data/pickle/types-entities.pkl", "rb") as f:
        types, entities = pickle.load(f)
    
    with open("Data/pickle/index.pkl", "rb") as f:
        index = pickle.load(f)

    print("Preparing entity retrieval")

    vectorizer = HashingVectorizer(alternate_sign=False, stop_words="english")

    bm25 = BM25_sklearn(index)
    ec = EntityCentric(np.array(types), np.array(entities), bm25, k=100)

    question_file = "Data/smart-dataset-master/datasets/DBpedia/smarttask_dbpedia_train.json"
    pred_file = "Data/prediction_train.json"

    with open(question_file, "r")  as read_f:
        data = json.load(read_f)

    print("Begin processing questions")
    with open(pred_file, "w") as write_f:
        # total = len(data)
        out = []
        # for i, q in enumerate(data):
        for q in tqdm.tqdm(data, total=len(data)):
            # if (i + 1) % (total // 1000) == 0:
            #     print(f"\r{round(100*(i/total), 1)}% processed...", end="")
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
        json.dump(out, write_f)
        
