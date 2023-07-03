# ref https://www.kaggle.com/code/vpkprasanna/bert-model-with-0-845-accuracy/notebook


import os
import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

words = stopwords.words("english")
lemma = nltk.stem.WordNetLemmatizer()

import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from transformers import BertModel

import random
from io import StringIO

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def set_seed(seed_value=3407):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


set_seed()

with open('./data/test.csv') as file:
    lines = [re.sub(
        r'([^,])"(\s*[^\n])',
        r'\1`"\2',
        line
    ) for line in file]
    test = pd.read_csv(
        StringIO(''.join(lines)),
        escapechar='`'
    )
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
MAX_LEN = 100


def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """

    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"won't", "will not ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'\n", " ", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"\'\xa0", " ", text)
    text = re.sub('\s+', ' ', text)
    text = ''.join(c for c in text if not c.isnumeric())

    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # for every sentence...

    for sent in data:
        # 'encode_plus will':
        # (1) Tokenize the sentence
        # (2) Add the `[CLS]` and `[SEP]` token to the start and end
        # (3) Truncate/Pad sentence to max length
        # (4) Map tokens to their IDs
        # (5) Create attention mask
        # (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,  # Max length to truncate/pad
            pad_to_max_length=True,  # pad sentence to max length
            return_attention_mask=True  # Return attention mask
        )
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []
    path_to_save = os.path.join('submissions', '{}.csv'.format("bert_0_4"))
    with open(path_to_save, 'w') as f:
        f.write('ImageID,Labels\n')

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        image_file, b_input_ids, b_attn_mask = batch
        b_input_ids, b_attn_mask = tuple([b_input_ids, b_attn_mask])[:2]
        b_input_ids = b_input_ids.to(device)
        b_attn_mask = b_attn_mask.to(device)

        # Compute label and save
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
            preds = logits.sigmoid() > 0.4
            with open(path_to_save, 'a') as f:
                for im_id, pred in zip(image_file, preds):
                    # Decode preds
                    lbl_idxs = [idx for idx, val in enumerate(pred) if val]
                    out = [str(i) for i in lbl_idxs]
                    string = f'{im_id},{" ".join(out)}\n'
                    f.write(string)

        all_logits.append(logits)
    print('Done! Predictions saved to {}'.format(path_to_save))


class BertClassifier(nn.Module):
    """
        Bert Model for classification Tasks.
    """

    def __init__(self, freeze_bert=False):
        """
        @param   bert: a BertModel object
        @param   classifier: a torch.nn.Module classifier
        @param   freeze_bert (bool): Set `False` to fine_tune the Bert model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of Bert, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 30, 20

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out))

        # Freeze the Bert Model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logit = self.classifier(last_hidden_state_cls)

        #         logits = self.sigmoid(logit)

        return logit


bert_classifier = BertClassifier(freeze_bert=False)
bert_classifier.load_state_dict(torch.load('./best_bert.pth'))

bert_classifier.to(device)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_id, test_inputs, test_masks):
        self.img_id = img_id
        self.test_inputs = test_inputs
        self.test_masks = test_masks

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        return self.img_id[idx], self.test_inputs[idx], self.test_masks[idx]


test["text"] = test["Caption"]
test["text"] = test["text"].apply(text_preprocessing)
img_id = test["ImageID"]

## Run preprocessing_for_bert on the test set
print('Tokenizing data...')
test_inputs, test_masks = preprocessing_for_bert(test.text)

# Create the DataLoader for our test set
test_dataset = TestDataset(img_id, test_inputs, test_masks)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)

bert_predict(bert_classifier, test_dataloader)
