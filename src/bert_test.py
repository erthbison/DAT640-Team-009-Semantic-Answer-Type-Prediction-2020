from round1_prep import question_target
import transformers
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import BertModel,BertTokenizer,AdamW,get_linear_schedule_with_warmup
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
import json
import numpy as np
RANDOM_SEED = 420

#! TO CHANGE PATHS FROM ABSOLUTE (THEY DO NOT WORK ON MY PC)
train_file = r"C:\Users\ziadr\Desktop\dat640\smart-dataset\DAT640-Team-009-Semantic-Answer-Type-Prediction-2020\Data\smart-dataset-master\datasets\DBpedia\smarttask_dbpedia_train.json"
test_file = r"C:\Users\ziadr\Desktop\dat640\smart-dataset\DAT640-Team-009-Semantic-Answer-Type-Prediction-2020\Data\smart-dataset-master\datasets\DBpedia\smarttask_dbpedia_test.json"


def get_resources(filename):
    #Returns a list of questions whose answers categories are resources + the list of resources
    questions,resource_list = [],[]
    with open(filename, "r") as f:
        data = json.load(f)
    for parsed in data:
        if parsed["category"] == "resource":
            questions.append(parsed["question"])
            resource_list.append(parsed["type"])
    return questions,resource_list

def resource_threshold(k,filename = train_file): #k is the threshold of occurences of a resource type
    #returns a list of resources that satisfy : occurences > k
    dico = {} #Dictionary mapping resource type -> number of occurences
    _,resource_lists = get_resources(filename)
    for ls in resource_lists: #We loop over the lists of resources
        for resource in ls: #We loop over every individual resource
            if resource in dico.keys():
                dico[resource] += 1
            else:
                dico[resource] = 1
    return [resource for resource in dico.keys() if dico[resource]>k]


targets = resource_threshold(25) #95 resources are present in this list


#Encoding labels
def label_binarizer(resource_list,targets = targets): #Takes a resource list and returns the encoded vector
    res = np.zeros((1,len(targets)))
    for resource in resource_list:
        index = targets.index(resource)
        res[0,index] = 1
    return res

#Decoding labels
def inverse_label(encoded_label, targets=targets):
    return [targets[i] for i in range(len(targets)) if encoded_label[0,i] == 1]

#TRAINING AND TESTING DATA

X_train,y_train = get_resources(train_file)
X_test,y_test = get_resources(test_file)

#This step is necessary for now, we will remove all the resources that are not present in our target resources 
#Training set
#This whole procedure only serves to REMOVE any resource type and corresponding question that is not present in our targets
for i in range(len(y_train)):
    ls = []
    for resource in y_train[i]:
        if resource in targets:
            ls.append(resource)
    y_train[i] = ls
ls1,ls2 = [],[]
for i in range(len(X_train)):
    if y_train[i] != []:
        ls1.append(X_train[i])
        ls2.append(y_train[i])
X_train,y_train = ls1,ls2
#Same procedure for test labels
for i in range(len(y_test)):
    ls = []
    for resource in y_test[i]:
        if resource in targets:
            ls.append(resource)
    y_test[i] = ls
ls1,ls2 = [],[]
for i in range(len(X_test)):
    if y_test[i] != []:
        ls1.append(X_test[i])
        ls2.append(y_test[i])
X_test,y_test = ls1,ls2


#Splitting into validation
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=RANDOM_SEED,shuffle=True)

#Encoding the labels
y_train,y_val = [label_binarizer(label) for label in y_train],[label_binarizer(label) for label in y_val]
y_test = [label_binarizer(label) for label in y_test]

class LabelDataset (Dataset):
    def __init__(self,quest,resources, tokenizer, max_len):
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
            add_special_tokens=True, # Add [CLS] [SEP]
            max_length= self.max_len,
            padding = 'max_length',
            return_token_type_ids= False,
            return_attention_mask= True, # Differentiates padded vs normal token
            truncation=True, # Truncate data beyond max length
            return_tensors = 'pt' # PyTorch Tensor format
          )
        
        input_ids = inputs['input_ids'].flatten()
        attn_mask = inputs['attention_mask'].flatten()
        
        return {
            'input_ids': input_ids ,
            'attention_mask': attn_mask,
            'label': torch.tensor(self.labels[item_idx], dtype=torch.float)
            
        }

class LabelDataModule (pl.LightningDataModule):
    
    def __init__(self,x_tr,y_tr,x_val,y_val,x_test,y_test,tokenizer,batch_size=16,max_token_len=200):
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
        self.train_dataset = LabelDataset(quest=self.tr_text, resources=self.tr_label, tokenizer=self.tokenizer,max_len = self.max_token_len)
        self.val_dataset  = LabelDataset(quest=self.val_text,resources=self.val_label,tokenizer=self.tokenizer,max_len = self.max_token_len)
        self.test_dataset  = LabelDataset(quest=self.test_text,resources=self.test_label,tokenizer=self.tokenizer,max_len = self.max_token_len)
        
        
    def train_dataloader(self):
        return DataLoader (self.train_dataset,batch_size = self.batch_size,shuffle = True , num_workers=2)

    def val_dataloader(self):
        return DataLoader (self.val_dataset,batch_size= 16)

    def test_dataloader(self):
        return DataLoader (self.test_dataset,batch_size= 16)

#BERT INITIALIZATION
BERT_MODEL_NAME = "bert-base-cased"
Bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

# Initialize the parameters that will be use for training
N_EPOCHS = 12
BATCH_SIZE = 32
MAX_LEN = 300
LR = 2e-05

Labeldata_module = LabelDataModule(X_train,y_train,X_val,y_val,X_test,y_test,Bert_tokenizer,BATCH_SIZE,MAX_LEN)
Labeldata_module.setup()
print(torch.cuda.device_count())
class LabelClassifier(pl.LightningModule):
    # Set up the classifier
    def __init__(self, n_classes=10, steps_per_epoch=None, n_epochs=3, lr=2e-5 ):
        super().__init__()

        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size,n_classes) # outputs = number of labels
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self,input_ids, attn_mask):
        output = self.bert(input_ids = input_ids ,attention_mask = attn_mask)
        output = self.classifier(output.pooler_output)
                
        return output
    
    
    def training_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        outputs = self(input_ids,attention_mask)
        loss = self.criterion(outputs,labels)
        self.log('train_loss',loss , prog_bar=True,logger=True)
        
        return {"loss" :loss, "predictions":outputs, "labels": labels }


    def validation_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        outputs = self(input_ids,attention_mask)
        loss = self.criterion(outputs,labels)
        self.log('val_loss',loss , prog_bar=True,logger=True)
        
        return loss

    def test_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        outputs = self(input_ids,attention_mask)
        loss = self.criterion(outputs,labels)
        self.log('test_loss',loss , prog_bar=True,logger=True)
        
        return loss
    
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters() , lr=self.lr)
        warmup_steps = self.steps_per_epoch//3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(optimizer,warmup_steps,total_steps)

        return [optimizer], [scheduler]

steps_per_epoch = len(X_train)//BATCH_SIZE
model = LabelClassifier(n_classes=10, steps_per_epoch=steps_per_epoch,n_epochs=N_EPOCHS,lr=LR)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',# monitored quantity
    filename='Label-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3, #  save the top 3 models
    mode='min', # mode of the monitored quantity  for optimization
)

trainer = pl.Trainer(max_epochs = N_EPOCHS , callbacks=[checkpoint_callback],accelerator='cpu')
trainer.fit(model, Labeldata_module)

