import json
import pickle
import torch
from helper_functions import  select_types
from DBpedia import get_types
from round1_prep import (extract_features, get_category_type, get_label,
                         question_target, test_file, train_file)
from pred_bert import predict
if __name__ == "__main__":
    print("Starting Part 1")
    print("Loading model")
    with open("model.pkl",'rb') as f:
        model = pickle.load(f)

    print("Preparing data")
    #Preparations
    ls_train,label_train = question_target(train_file)
    ls_test,label_test = question_target(test_file)
    X_train,X_test = extract_features(ls_train, ls_test)
    label_test = [get_label(x) for x in label_test]

    #PREDICTIONS
    #LIST OF LABELS FOLLOWING THE MAPPING OF f from round1_prep where boolean -> 0, number -> 1 ...
    print("Performing prediction")
    results = model.predict(X_test)

    print("Completed Part 1")
    print("Starting Part 2")

    print("Loading data")
    with open(test_file, "r")  as read_f:
        questions = json.load(read_f)

    #ec, vectorizer = load_entity_retrieval(150)

    out = []
    total = len(results)
    dico = get_types("dbpedia_2016-10.nt") #Path has to be changed 
    print("Starting prediction")
    model = torch.load("./models")
    for i, (category, question) in enumerate(zip(results, questions)):
        category, q_type = get_category_type(category)
        if (i + 1) % (total // 1000) == 0:
            print(f"\r{round(100*(i/total), 1)}% processed...", end="")
        if category == "resource":
            #query = vectorizer.transform([question["question"]])
            #predicted = ec.Score(query)
            type_name = predict(question["question"],model=model)
            if type_name == "dbo:Location":
                type_name = "dbo:Place"
            predicted = [(0,dico[type_name[4:]])] #BERT model version
            pred = {
                "id": question["id"],
                "category": category,
                "type": select_types(predicted)
            }
        else:
            pred = {
                "id": question["id"],
                "category": category,
                "type": [q_type],
            }
        out.append(pred)

    print("Completed Part 2")

    with open("advanced_predictions.json", "w") as f:
        json.dump(out, f)
