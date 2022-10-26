import pickle
from sklearn import svm
from round1_prep import question_target,f,extract_features
model = pickle.load(open("./model",'rb'))

#Preparations
ls_train,label_train = question_target(r"C:\Users\ziadr\Desktop\dat640\smart-dataset\DAT640-Team-009-Semantic-Answer-Type-Prediction-2020\Data\smart-dataset-master\datasets\DBpedia\smarttask_dbpedia_train.json")

ls_test,label_test = question_target(r"C:\Users\ziadr\Desktop\dat640\smart-dataset\DAT640-Team-009-Semantic-Answer-Type-Prediction-2020\Data\smart-dataset-master\datasets\DBpedia\smarttask_dbpedia_test.json")
X_train,X_test = extract_features(ls_train,ls_test)
label_test = [f(x) for x in label_test]

#PREDICTIONS
#LIST OF LABELS FOLLOWING THE MAPPING OF f from round1_prep where boolean -> 0, number -> 1 ...
results = model.predict(X_test)
