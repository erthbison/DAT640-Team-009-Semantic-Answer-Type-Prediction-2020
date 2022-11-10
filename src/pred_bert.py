import numpy as np
import torch
from transformers import BertForSequenceClassification,BertTokenizer 
from test_bert2 import preprocessing,targets
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#Loading model
model = torch.load("./models")


new_sentence = "What is the seat of Frankfurter Allgemeine Zeitung, that is adjacent to Wetteraukreis?"

# We need Token IDs and Attention Mask for inference on the new sentence
test_ids = []
test_attention_mask = []
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case = True
    )
# Apply the tokenizer
encoding = preprocessing(new_sentence, tokenizer)

# Extract IDs and Attention Mask
test_ids.append(encoding['input_ids'])
test_attention_mask.append(encoding['attention_mask'])
test_ids = torch.cat(test_ids, dim = 0)
test_attention_mask = torch.cat(test_attention_mask, dim = 0)

# Forward pass, calculate logit predictions
with torch.no_grad():
  output = model(test_ids.to(device), token_type_ids = None, attention_mask = test_attention_mask.to(device))

prediction = targets[np.argmax(output.logits.cpu().numpy()).flatten().item()]

print('Input Sentence: ', new_sentence)
print('Predicted Class: ', prediction)