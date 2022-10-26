from typing import Tuple, Union

import pickle
from sklearn import svm
from round1_prep import question_target,f,extract_features

from baseline import load_entity_retrieval, select_types

import json

model = pickle.load(open("./model",'rb'))

train_file = r"Data\smart-dataset-master\datasets\DBpedia\smarttask_dbpedia_train.json"
test_file = r"Data\smart-dataset-master\datasets\DBpedia\smarttask_dbpedia_test.json"

#Preparations
ls_train,label_train = question_target(train_file)
ls_test,label_test = question_target(test_file)
X_train,X_test = extract_features(ls_train, ls_test)
label_test = [f(x) for x in label_test]

#PREDICTIONS
#LIST OF LABELS FOLLOWING THE MAPPING OF f from round1_prep where boolean -> 0, number -> 1 ...
results = model.predict(X_test)

print("Completed Part 1")

def get_category_type(x) -> Tuple[str, Union[str, None]]:
    # A mapping from a part 1 label to a category and a type
    if x == 2:
        return "literal", "string"
    if x == 3:
        return "literal", "date"
    if x == 0:
        return "boolean", "boolean"
    if x == 4:
        return "resource", None
    if x == 1:
        return "literal", "number"

with open(test_file, "r")  as read_f:
    data = json.load(read_f)

ec, vectorizer = load_entity_retrieval(150)

out = []
total = len(results)
assert len(results) == len(data)

print("Starting Part 2")
for i, (category, question) in enumerate(zip(results, data)):
    category, q_type = get_category_type(category)
    if (i + 1) % (total // 1000) == 0:
        print(f"\r{round(100*(i/total), 1)}% processed...", end="")
    if category == "resource":
        query = vectorizer.transform([question["question"]])
        predicted = ec.Score(query)
        pred = {
            "id": question["id"],
            "category": category,
            "type": select_types(predicted)
        }
    else:
        # Just fill in this with the true data just so we get a measure
        pred = {
            "id": question["id"],
            "category": category,
            "type": [q_type],
        }
    out.append(pred)

print("Completed Part 2")

with open("Data/baseline_predictions.json", "w") as f:
    json.dump(out, f)
