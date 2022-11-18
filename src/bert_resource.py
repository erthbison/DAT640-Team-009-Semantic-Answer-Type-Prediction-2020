import os
from round1_prep import question_target
import torch
from tqdm import trange
from torch.utils.data import TensorDataset
from sklearn import metrics
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from sklearn.model_selection import train_test_split
import json
import numpy as np

from test_bert2 import b_metrics
# Generate a random seed
RANDOM_SEED = __import__("random").randint(1, 1000)
# If the seed failed to be set, we set it manually
if RANDOM_SEED == None:
	RANDOM_SEED = 420
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Datasets filepaths
train_file = ".\Data\smart-dataset-master\datasets\DBpedia\smarttask_dbpedia_train.json"
test_file = ".\Data\smart-dataset-master\datasets\DBpedia\smarttask_dbpedia_test.json"


def get_resources(filename: str) -> tuple[list[str], list[list[str]]]:
	"""
	Read through the file and the list of resources for each question

	Args:
	----
		filename (str): path to the file

	Returns:
	-------
		tuple: a list of questions whose answers categories are resources, and the list of said resources
	"""
	questions, resource_list = [], []
	with open(filename, "r") as f:
		data = json.load(f)
	# We loop over the questions in the dataset and we keep only the ones whose answers are resources
	for parsed in data:
		if parsed["category"] == "resource" and parsed["question"] != None:
			questions.append(parsed["question"])
			resource_list.append(parsed["type"])
	return questions, resource_list


def resource_threshold(k: int, filename: str = train_file) -> list[str]:
	"""
	Return the list of resources that appear at least k times in the training set

	Args:
	----
		k (int): threshold value of ocurrences of a resource
		filename (str, optional): path to the file. Defaults to train_file.

	Returns:
	-------
		list(str): list of resources that appear more than k times in the dataset
	"""
	dictionary = {}
	_, resource_lists = get_resources(filename)
	# We loop over the list of resources for each question and we count the number of times each resource appears
	for ls in resource_lists:
		for resource in ls:
			if resource in dictionary.keys():
				dictionary[resource] += 1
			else:
				dictionary[resource] = 1
	return [resource for resource in dictionary.keys() if dictionary[resource] > k]


targets = resource_threshold(20)  # We choose k=20 for now

# Encoding labels => every label is assigned to its index in targets


def encoding_label(label: str, targets: list[str] = targets) -> int:
	"""
	Encode the label of a question

	Args:
	----
		label (str): label of a question
		targets (list(str), optional): list of resources. Defaults to targets.

	Returns:
	-------
		int: index of the label in the list of resources
	"""
	return targets.index(label)
# Decoding labels => targets[label]

# TRAINING AND TESTING DATA
# We start from question -> [resource list] to question, question, question -> resource1, resource2, resource3
# for the training process. We do the same for the testing process.
# So we have three instances of the same question but assigned to its different resources individually


train_questions, train_labels = get_resources(train_file)
test_questions, test_labels = get_resources(test_file)
X_train, y_train = [], []
X_test, y_test = test_questions, test_labels

# question, question, question -> resource1, resource2, resource3
for i in range(len(train_questions)):
	for label in train_labels[i]:
		if label in targets:
			X_train.append(train_questions[i])
			y_train.append(encoding_label(label))
for i in range(len(test_questions)):
	for label in test_labels[i]:
		if label in targets:
			X_test.append(test_questions[i])
			y_test.append(encoding_label(label))
# ^ At the end of this procedure we have our train and test datasets

# Tokenizer
tokenizer = BertTokenizer.from_pretrained(
	'bert-base-uncased',
	do_lower_case=True
)
# Preprocessing function


def preprocessing(input_text: str, tokenizer: BertTokenizer) -> BertTokenizer:
	'''
	Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
		  - input_ids: list of token ids
		  - token_type_ids: list of token type ids
		  - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
	'''
	return tokenizer.encode_plus(
		input_text,
		add_special_tokens=True,
		max_length=32,
		pad_to_max_length=True,
		return_attention_mask=True,
		return_tensors='pt'
	)


# Procedure for training only (for now)
token_id, attention_masks = [], []

for tr_question in X_train:
	encoding_dict = preprocessing(tr_question, tokenizer)
	token_id.append(encoding_dict['input_ids'])
	attention_masks.append(encoding_dict['attention_mask'])

# Convert to tensors
token_id = torch.cat(token_id, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
y_train = torch.tensor(y_train)

# Splitting
val_ratio = 0.2
batch_size = 16

# Create TensorDatasets for train and validation sets
train_idx, val_idx = train_test_split(
	np.arange(len(y_train)),
	test_size=val_ratio,
	shuffle=True,
	stratify=y_train)

train_set = TensorDataset(token_id[train_idx],
						  attention_masks[train_idx],
						  y_train[train_idx])

validation_set = TensorDataset(token_id[val_idx],
							   attention_masks[val_idx],
							   y_train[val_idx])

train_dataloader = DataLoader(
	train_set,
	sampler=RandomSampler(train_set),
	batch_size=batch_size
)

validation_dataloader = DataLoader(
	validation_set,
	sampler=SequentialSampler(validation_set),
	batch_size=batch_size
)


# TRAINING
# Load the BertForSequenceClassification model in order to fine-tune it
model = BertForSequenceClassification.from_pretrained(
	'bert-base-uncased',
	num_labels=len(targets),
	output_attentions=False,
	output_hidden_states=False,
)

optimizer = torch.optim.AdamW(model.parameters(),
							  lr=5e-5,
							  eps=1e-08
							  )

# Run on GPU
model.cuda()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 5

if __name__ == "__main__":
	# Training part, can be skipped if the model is already trained (comment the following lines)

	# If there is no model saved, we train the model
	if not os.path.exists('./models'):
		print("Training started")
		for _ in trange(epochs, desc='Epoch'):

			# ========== Training ==========

			# Set model to training mode
			model.train()

			# Tracking variables
			tr_loss = 0
			nb_tr_examples, nb_tr_steps = 0, 0

			for step, batch in enumerate(train_dataloader):
				batch = tuple(t.to(device) for t in batch)
				b_input_ids, b_input_mask, b_labels = batch
				optimizer.zero_grad()
				# Forward pass
				train_output = model(b_input_ids,
									 token_type_ids=None,
									 attention_mask=b_input_mask,
									 labels=b_labels)
				# Backward pass
				train_output.loss.backward()
				optimizer.step()
				# Update tracking variables
				tr_loss += train_output.loss.item()
				nb_tr_examples += b_input_ids.size(0)
				nb_tr_steps += 1

			# ========== Validation ==========

			# Set model to evaluation mode
			model.eval()

			# Tracking variables
			val_accuracy = []
			val_precision = []
			val_recall = []
			val_specificity = []

			for batch in validation_dataloader:
				batch = tuple(t.to(device) for t in batch)
				b_input_ids, b_input_mask, b_labels = batch
				with torch.no_grad():
					# Forward pass
					eval_output = model(b_input_ids,
										token_type_ids=None,
										attention_mask=b_input_mask)
				logits = eval_output.logits.detach().cpu().numpy()
				label_ids = b_labels.to('cpu').numpy()
				# Calculate validation metrics
				b_accuracy, b_precision, b_recall, b_specificity = b_metrics(
					logits, label_ids)
				val_accuracy.append(b_accuracy)
				# Update precision only when (tp + fp) !=0; ignore nan
				if b_precision != 'nan':
					val_precision.append(b_precision)
				# Update recall only when (tp + fn) !=0; ignore nan
				if b_recall != 'nan':
					val_recall.append(b_recall)
				# Update specificity only when (tn + fp) !=0; ignore nan
				if b_specificity != 'nan':
					val_specificity.append(b_specificity)

			print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
			print(
				'\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
			print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(
				val_precision)) if len(val_precision) > 0 else '\t - Validation Precision: NaN')
			print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(
				val_recall)) if len(val_recall) > 0 else '\t - Validation Recall: NaN')
			print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity)/len(
				val_specificity)) if len(val_specificity) > 0 else '\t - Validation Specificity: NaN')
		torch.save(model, "./models")

	# EVALUATION
	model = torch.load("./models")
	model.eval()
	acc = 0
	preds = []

	# Prediction on test set and evaluation metrics calculation (accuracy, precision, recall, specificity)
	for i in range(0, len(X_test), 16):
		test_ids, test_attention_masks = [], []
		for j in range(16):
			if i+j >= len(X_test):
				break
			encoding = preprocessing(X_test[i+j], tokenizer=tokenizer)
			test_ids.append(encoding["input_ids"])
			test_attention_masks.append(encoding["attention_mask"])
		test_ids = torch.cat(test_ids, dim=0)
		test_attention_masks = torch.cat(test_attention_masks, dim=0)
		with torch.no_grad():
			# Forward pass
			eval_output = model(test_ids.to(device),
								token_type_ids=None,
								attention_mask=test_attention_masks.to(device))
		logits = eval_output.logits.detach().cpu().numpy()
		preds.append(logits)

	for i in range(len(preds)):
		for j in range(len(preds[i])):
			predicted = np.argmax(preds[i][j]).flatten().item()
			if targets[predicted] in y_test[i*16+j]:
				acc += 1
	print(acc/len(y_test))
