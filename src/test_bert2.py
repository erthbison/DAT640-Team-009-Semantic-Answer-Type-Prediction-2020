import torch
from tqdm import trange
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from sklearn.model_selection import train_test_split
import json
import numpy as np
# Generate a random seed
RANDOM_SEED = __import__("random").randint(1, 1000)
# If the seed failed to be set, we set it manually
if RANDOM_SEED == None:
	RANDOM_SEED = 420
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#! TO CHANGE PATHS FROM ABSOLUTE (THEY DO NOT WORK ON MY PC)
train_file = ".\Data\smart-dataset-master\datasets\DBpedia\smarttask_dbpedia_train.json"
test_file = ".\Data\smart-dataset-master\datasets\DBpedia\smarttask_dbpedia_test.json"


# METRICS
def b_tp(preds, labels):
	'''Returns True Positives (TP): count of correct predictions of actual class 1'''
	return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])


def b_fp(preds, labels):
	'''Returns False Positives (FP): count of wrong predictions of actual class 1'''
	return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])


def b_tn(preds, labels):
	'''Returns True Negatives (TN): count of correct predictions of actual class 0'''
	return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])


def b_fn(preds, labels):
	'''Returns False Negatives (FN): count of wrong predictions of actual class 0'''
	return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])


def b_metrics(preds, labels):
	'''
	Returns the following metrics:
	  - accuracy    = (TP + TN) / N
	  - precision   = TP / (TP + FP)
	  - recall      = TP / (TP + FN)
	  - specificity = TN / (TN + FP)
	'''
	preds = np.argmax(preds, axis=1).flatten()
	labels = labels.flatten()
	tp = b_tp(preds, labels)
	tn = b_tn(preds, labels)
	fp = b_fp(preds, labels)
	fn = b_fn(preds, labels)
	b_accuracy = (tp + tn) / len(labels)
	b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
	b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
	b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
	return b_accuracy, b_precision, b_recall, b_specificity

# END OF METRICS


def get_resources(filename):
	# Returns a list of questions whose answers categories are resources + the list of resources
	questions, resource_list = [], []
	with open(filename, "r") as f:
		data = json.load(f)
	for parsed in data:
		if parsed["category"] == "resource" and parsed["question"] != None:
			questions.append(parsed["question"])
			resource_list.append(parsed["type"])
	return questions, resource_list


# k is the threshold of occurences of a resource type
def resource_threshold(k, filename=train_file):
	# returns a list of resources that satisfy : occurences > k
	dico = {}  # Dictionary mapping resource type -> number of occurences
	_, resource_lists = get_resources(filename)
	for ls in resource_lists:  # We loop over the lists of resources
		for resource in ls:  # We loop over every individual resource
			if resource in dico.keys():
				dico[resource] += 1
			else:
				dico[resource] = 1
	return [resource for resource in dico.keys() if dico[resource] > k]


targets = resource_threshold(20)  # We choose k=20 for now

# Encoding labels => every label is assigned to its index in targets


def encoding_label(label, targets=targets):
	return targets.index(label)
# Decoding labels => targets[label]

# TRAINING AND TESTING DATA
# We start from question->[resource list] to question,question,question->resource1,resource2,resource3 for the training process
# So we have three instances of the same question but assigned to its different resources individually


# question->[resource list]
train_questions, train_labels = get_resources(train_file)
test_questions, test_labels = get_resources(test_file)
X_train, y_train = [], []
X_test, y_test = [], []

# question,question,question->resource1,resource2,resource3
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


def preprocessing(input_text, tokenizer):
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


token_id = torch.cat(token_id, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
y_train = torch.tensor(y_train)

# Splitting
val_ratio = 0.2
batch_size = 16

train_idx, val_idx = train_test_split(
	np.arange(len(y_train)),
	test_size=val_ratio,
	shuffle=True,
	stratify=y_train)

train_set = TensorDataset(token_id[train_idx],
						  attention_masks[train_idx],
						  y_train[train_idx])

val_set = TensorDataset(token_id[val_idx],
						attention_masks[val_idx],
						y_train[val_idx])
train_dataloader = DataLoader(
	train_set,
	sampler=RandomSampler(train_set),
	batch_size=batch_size
)

validation_dataloader = DataLoader(
	val_set,
	sampler=SequentialSampler(val_set),
	batch_size=batch_size
)

# TRAINING
# Load the BertForSequenceClassification model
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
	print("Commented out training")
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
