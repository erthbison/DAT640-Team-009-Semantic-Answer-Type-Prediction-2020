import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import json
import numpy as np
# Generate a random seed
RANDOM_SEED = __import__("random").randint(1, 1000)
# If the seed failed to be set, we set it manually
if RANDOM_SEED == None:
	RANDOM_SEED = 420

train_file = ".\Data\smart-dataset-master\datasets\DBpedia\smarttask_dbpedia_train.json"
test_file = ".\Data\smart-dataset-master\datasets\DBpedia\smarttask_dbpedia_test.json"


def get_resources(filename: str) -> tuple(list(str), list(list(str))):
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
	for parsed in data:
		if parsed["category"] == "resource":
			questions.append(parsed["question"])
			resource_list.append(parsed["type"])
	return questions, resource_list


def resource_threshold(k: int, filename: str = train_file) -> list(str):
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


targets = resource_threshold(25)  # 95 resources are present in this list


# Encoding labels
# Takes a resource list and returns the encoded vector
def label_binarizer(resource_list: list, targets: list = targets) -> list:
	"""
	Takes a resource list and returns the encoded vector

	Args:
	----
		resource_list (list): list of all resources
		targets (list, optional): list of resources that appear more than a certain amount in the dataset. Defaults to targets (25).

	Returns:
	-------
		list: encoded vector
	"""
	res = np.zeros((1, len(targets)))
	for resource in resource_list:
		index = targets.index(resource)
		res[0, index] = 1
	return res

# Decoding labels

def inverse_label(encoded_label: list, targets: list = targets) -> list:
	"""
	Takes an encoded label and returns the list of resources

	Args:
	----
		encoded_label (list): encoded label
		targets (list, optional): list of resources that appear more than a certain amount in the dataset. Defaults to targets (25).

	Returns:
	-------
		list: list of resources
	"""
	return [targets[i] for i in range(len(targets)) if encoded_label[0, i] == 1]

# TRAINING AND TESTING DATA


X_train, y_train = get_resources(train_file)
X_test, y_test = get_resources(test_file)

# This step is necessary for now, we will remove all the resources that are not present in our target resources
# Training set
# This whole procedure only serves to REMOVE any resource type and corresponding question that is not present in our targets
for i in range(len(y_train)):
	ls = []
	for resource in y_train[i]:
		if resource in targets:
			ls.append(resource)
	y_train[i] = ls
ls1, ls2 = [], []
for i in range(len(X_train)):
	if y_train[i] != []:
		ls1.append(X_train[i])
		ls2.append(y_train[i])
X_train, y_train = ls1, ls2
# Same procedure for test labels
for i in range(len(y_test)):
	ls = []
	for resource in y_test[i]:
		if resource in targets:
			ls.append(resource)
	y_test[i] = ls
ls1, ls2 = [], []
for i in range(len(X_test)):
	if y_test[i] != []:
		ls1.append(X_test[i])
		ls2.append(y_test[i])
X_test, y_test = ls1, ls2


# Splitting into validation
X_train, X_val, y_train, y_val = train_test_split(
	X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)

# Encoding the labels
y_train, y_val = [label_binarizer(label) for label in y_train], [
	label_binarizer(label) for label in y_val]
y_test = [label_binarizer(label) for label in y_test]


class LabelDataset (Dataset):
	def __init__(self, quest, resources, tokenizer, max_len):
		self.tokenizer = tokenizer
		self.text = quest
		self.labels = resources
		self.max_len = max_len

	def __len__(self):
		return len(self.text)

	def __getitem__(self, item_idx):
		text = self.text[item_idx]
		inputs = self.tokenizer.encode_plus(
			text,
			None,
			add_special_tokens=True,  # Add [CLS] [SEP]
			max_length=self.max_len,
			padding='max_length',
			return_token_type_ids=False,
			return_attention_mask=True,  # Differentiates padded vs normal token
			truncation=True,  # Truncate data beyond max length
			return_tensors='pt'  # PyTorch Tensor format
		)

		input_ids = inputs['input_ids'].flatten()
		attn_mask = inputs['attention_mask'].flatten()

		return {
			'input_ids': input_ids,
			'attention_mask': attn_mask,
			'label': torch.tensor(self.labels[item_idx], dtype=torch.float)

		}


class LabelDataModule (pl.LightningDataModule):
	"""
	DataModule for the label classification task

	Attributes:
	----------
		tr_text : str
			list of questions in the training set
		tr_labels : list
			list of resources in the training set
		val_text : str
			list of questions in the validation set
		val_labels : list
			list of resources in the validation set
		test_text : str
			list of questions in the test set
		test_labels : list
			list of resources in the test set
		tokenizer : transformers.PreTrainedTokenizer
			tokenizer used to tokenize the questions
		batch_size : int
			batch size used for training
		max_token_len : int
			maximum length of a tokenized question

	Methods:
	-------
		setup(stage=None): splits the data into training, validation and test sets
		train_dataloader(): returns the training dataloader
		val_dataloader(): returns the validation dataloader
		test_dataloader(): returns the test dataloader
	"""

	def __init__(self, x_tr, y_tr, x_val, y_val, x_test, y_test, tokenizer, batch_size=16, max_token_len=200):
		super().__init__()
		self.tr_text = x_tr
		self.tr_label = y_tr
		self.val_text = x_val
		self.val_label = y_val
		self.test_text = x_test
		self.test_label = y_test
		self.tokenizer = tokenizer
		self.batch_size = batch_size
		self.max_token_len = max_token_len

	def setup(self):
		self.train_dataset = LabelDataset(
			quest=self.tr_text, resources=self.tr_label, tokenizer=self.tokenizer, max_len=self.max_token_len)
		self.val_dataset = LabelDataset(
			quest=self.val_text, resources=self.val_label, tokenizer=self.tokenizer, max_len=self.max_token_len)
		self.test_dataset = LabelDataset(
			quest=self.test_text, resources=self.test_label, tokenizer=self.tokenizer, max_len=self.max_token_len)

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=16)

	def test_dataloader(self):
		return DataLoader(self.test_dataset, batch_size=16)


# BERT INITIALIZATION
BERT_MODEL_NAME = "bert-base-cased"
Bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

# Initialize the parameters that will be use for training
N_EPOCHS = 12
BATCH_SIZE = 32
MAX_LEN = 300
LR = 2e-05

Labeldata_module = LabelDataModule(
	X_train, y_train, X_val, y_val, X_test, y_test, Bert_tokenizer, BATCH_SIZE, MAX_LEN)
Labeldata_module.setup()
print(torch.cuda.device_count())


class LabelClassifier(pl.LightningModule):
	"""
	LightningModule for the label classification task

	Attributes:
	----------
		bert : transformers.BertModel
			BERT model used for the classification task
		dropout : torch.nn.Dropout
			dropout layer used to prevent overfitting
		classifier : torch.nn.Linear
			linear layer used to classify the resources
		step_per_epoch : int
			number of steps per per passes
		n_epochs : int
			number of passes through the training set
		lr : float
			learning rate used for training
		criterion : torch.nn.BCELoss
			loss function used for training

	Methods:
	-------
		forward(input_ids, attention_mask): returns the output of the classifier
		training_step(batch, batch_idx): returns the loss during training
		validation_step(batch, batch_idx): returns the loss during validation
		test_step(batch, batch_idx): returns the loss during testing
		configure_optimizers(): returns the optimizer used for training
	"""
	# Set up the classifier
	def __init__(self, n_classes=10, steps_per_epoch=None, n_epochs=3, lr=2e-5):
		super().__init__()

		self.bert = BertModel.from_pretrained(
			BERT_MODEL_NAME, return_dict=True)
		# outputs = number of labels
		self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
		self.steps_per_epoch = steps_per_epoch
		self.n_epochs = n_epochs
		self.lr = lr
		self.criterion = nn.BCEWithLogitsLoss()

	def forward(self, input_ids, attn_mask):
		output = self.bert(input_ids=input_ids, attention_mask=attn_mask)
		output = self.classifier(output.pooler_output)

		return output

	def training_step(self, batch, batch_idx):
		input_ids = batch['input_ids']
		attention_mask = batch['attention_mask']
		labels = batch['label']

		outputs = self(input_ids, attention_mask)
		loss = self.criterion(outputs, labels)
		self.log('train_loss', loss, prog_bar=True, logger=True)

		return {"loss": loss, "predictions": outputs, "labels": labels}

	def validation_step(self, batch, batch_idx):
		input_ids = batch['input_ids']
		attention_mask = batch['attention_mask']
		labels = batch['label']

		outputs = self(input_ids, attention_mask)
		loss = self.criterion(outputs, labels)
		self.log('val_loss', loss, prog_bar=True, logger=True)

		return loss

	def test_step(self, batch, batch_idx):
		input_ids = batch['input_ids']
		attention_mask = batch['attention_mask']
		labels = batch['label']

		outputs = self(input_ids, attention_mask)
		loss = self.criterion(outputs, labels)
		self.log('test_loss', loss, prog_bar=True, logger=True)

		return loss

	def configure_optimizers(self):
		optimizer = AdamW(self.parameters(), lr=self.lr)
		warmup_steps = self.steps_per_epoch//3
		total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

		scheduler = get_linear_schedule_with_warmup(
			optimizer, warmup_steps, total_steps)

		return [optimizer], [scheduler]

# Initialize the classifier and the trainer in order to train the model
steps_per_epoch = len(X_train)//BATCH_SIZE
model = LabelClassifier(
	n_classes=10, steps_per_epoch=steps_per_epoch, n_epochs=N_EPOCHS, lr=LR)

checkpoint_callback = ModelCheckpoint(
	monitor='val_loss',  # monitored quantity
	filename='Label-{epoch:02d}-{val_loss:.2f}',
	save_top_k=3,  # save the top 3 models
	mode='min',  # mode of the monitored quantity  for optimization
)

trainer = pl.Trainer(max_epochs=N_EPOCHS, callbacks=[
					 checkpoint_callback], accelerator='cpu')
trainer.fit(model, Labeldata_module)
