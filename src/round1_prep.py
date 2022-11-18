import json
import pickle
from typing import List, Tuple, Union

from numpy import ndarray
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

# File location for training and test file
train_file = ".\Data\smart-dataset-master\datasets\DBpedia\smarttask_dbpedia_train.json"
test_file = ".\Data\smart-dataset-master\datasets\DBpedia\smarttask_dbpedia_test.json"


def target_type(json_object: dict) -> str:
	"""Find the target type for a question object.

	The ML predictor is responsible for identifying the categories as well as the types for the Literal and the boolean category. 
	We therefore use the types as the target for these categories. 
	For questions where the category is resource Part 2 will be used to identify the type, so the category will be used as label. 
	"""
	# this function encapsulates the types to facilitate the ML process
	# It returns the target type of the json_object
	#boolean, string, date, number or resource
	if json_object["category"] == "resource":
		return json_object["category"]
	else:
		return json_object["type"][0]


def question_target(filename: str) -> Tuple[List[str], List[str]]:
	"""
	Create a list of preprocessed questions and a list of targets.


	The i-th question correspond to the i-th target.    
	"""
	# Takes the dataset and generates a list of [question,type]
	questions, targets = [], []
	with open(filename, "r") as f:
		data = json.load(f)
	for parsed in data:
		question, target = parsed["question"], target_type(parsed)
		questions.append(question or "")
		targets.append(target)
	return questions, targets


def get_label(x: str) -> int:
	"""Maps from a target type to an integer labels used by the ML algorithm. 

	The target type is either one of the three Literal types, the Boolean type or resource. 
	See the target_type() function for a more detailed description
	"""
	if x == "string":
		return 2
	if x == "date":
		return 3
	if x == "boolean":
		return 0
	if x == "resource":
		return 4
	if x == "number":
		return 1


def get_category_type(x: int) -> Tuple[str, Union[str, None]]:
	"""Maps from a ML label to a category and a type.

	If the category is resource the type is None, since the ML does not find the types of resources"""
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


def extract_features(
	train_dataset: List[str], test_dataset: List[str]
) -> Union[Tuple[ndarray, ndarray], Tuple[List[float], List[float]]]:
	"""Extracts feature vectors from a preprocessed train and test datasets.

	Args:
		train_dataset: List of strings, each consisting of the preprocessed
			email content.
		test_dataset: List of strings, each consisting of the preprocessed
			email content.

	Returns:
		A tuple of of two lists. The lists contain extracted features for 
		  training and testing dataset respectively.
	"""
	# This function will return the features to train the model on
	vectorizer = TfidfVectorizer()
	# We have to provide the union of both vocabularies in order to train
	X = vectorizer.fit_transform(train_dataset+test_dataset)
	X_train, X_test = X[:len(train_dataset), ], X[len(train_dataset):, ]
	return X_train, X_test


if __name__ == "__main__":
	# Prepare questions and labels for train and test data
	print("Preprocessing data")
	ls_train, label_train = question_target(train_file)
	ls_test, label_test = question_target(test_file)

	X_train, X_test = extract_features(ls_train, ls_test)
	label_train, label_test = [get_label(x) for x in label_train], [
		get_label(x) for x in label_test]

	# Instantiate and train ML model
	print("Training model")
	clf = svm.SVC()
	clf.fit(X_train, label_train)

	# Load model
	print("Saving model")
	with open("./Data/pickle/model.pkl", "wb") as f:
		pickle.dump(clf, f)
