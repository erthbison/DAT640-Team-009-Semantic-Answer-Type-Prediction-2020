
import pickle
import json

import numpy as np

from rank_bm25 import BM25Okapi

from retrieval_models import BM25
from ir import EntityCentric
from preprocess_data import preprocess, InvertedIndex, FieldedInvertedIndex

if __name__ == "__main__":

    with open("types-entities.pkl", "rb") as f:
        types, entities = pickle.load(f)
    # with open("Data/pickle/entity-index.pkl", "rb") as f:
    #     index = pickle.load(f)
    
    with open("Data/pickle/entity-term-list.pkl", "rb") as f:
        index = pickle.load(f)

    print("Preparing entity retrieval")
    # bm25 = BM25(index, b=0.75, k1=1.20)
    bm25 = BM25Okapi(index)
    ec = EntityCentric(np.array(types), np.array(entities), bm25, k=100)

    question_file = "Data/smart-dataset-master/datasets/DBpedia/smarttask_dbpedia_test.json"
    pred_file = "TEST.json"

    with open(question_file, "r")  as read_f:
        data = json.load(read_f)

    print("Begin processing questions")
    with open(pred_file, "w") as write_f:
        total = len(data)
        for i, q in enumerate(data):
            if (i + 1) % (total // 1000) == 0:
                print(f"\r{round(100*(i/total), 1)}% processed...", end="")
            if q["question"] and q["category"] == "resource":
                query = preprocess(q["question"])
                predicted = ec.Score(query)
                pred = {
                    "id": q["id"],
                    "category": q["category"],
                    "type": [(score, type.name) for score, type in predicted if score > 0]
                }
                json.dump(pred, write_f)
                write_f.flush()