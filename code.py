# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:45:09 2020

@author: Xinyi
"""

# pip install transformers

import torch
SEED = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DATA = 'sample_data/sub_train_others.csv'
VALID_DATA = 'sample_data/sub_dev_others.csv'
TEST_DATA = 'sample_data/test_others.csv'
RESULT_FILE = 'sample_data/result_test_others.csv'

MAX_LEN = 200
BATCH_SIZE = 16
LEARNING_RATE = 1e-6
ADAM_EPS = 1e-8
WARMUP_PROPORTION = 0.25
WEIGHT_DECAY = 0.01
NUM_EPOCH = 2
DROPOUT = 0.01
LABEL_LIST = [0, 1]
LABEL_SIZE = len(LABEL_LIST)*p

path = ''
DATA_DIR = path+"sample_data/"


#--------------------------------------Logger-----------------------------------------------#
import torch.nn.functional as F
import numpy as np
import logging
import csv
from logging import handlers
from torch.utils.data import DataLoader

def init_logger(filename, when='D', backCount=3,
                fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
    logger = logging.getLogger(filename)
    format_str = logging.Formatter(fmt)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
    th.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(th)
    return logger

logger = init_logger(filename=LOG_FILE)

# ----------------------------- utils --------------------------------------#
import csv
import pandas as pd
from torch.utils.data import TensorDataset
import random
import numpy as np


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class MyProcessor(object):
    """Processor for the racism data set."""

    def get_examples(self, prefix):
        """Gets the list of labels for this data set."""
        return self._create_examples(
            self._read_csv(DATA_DIR, FILENAME[prefix]), prefix)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _read_csv(cls, data_dir, filename, quotechar=None):
        """Reads a tab separated value file."""
        list_df = []
        for file in filename:
            list_df.append(pd.read_csv(os.path.join(data_dir, file), lineterminator='\n', sep=","))
        df = pd.concat(list_df)
        df = df.reset_index()
        lines = []
        for i in range(len(df)):
          lines.append([df["query1"][i], df["query2"][i], df["label"][i]])

        return lines

def _truncate_seq_pair(tokens_a, tokens_b, max_length: int):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, tokenizer,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (1 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(LABEL_LIST)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
            
        tokens_a, tokens_b = tokenizer.tokenize(example.text_a), tokenizer.tokenize(example.text_b)
        _truncate_seq_pair(tokens_a, tokens_b, MAX_LEN - 3)
        
        tokens = []
        segment_ids = []

        tokens.append(cls_token)
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append(sep_token)
        segment_ids.append(0)

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append(sep_token)
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)
        while len(input_ids) < MAX_LEN:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            tokens.append(pad_token)
            
        assert len(input_ids) == MAX_LEN
        assert len(input_mask) == MAX_LEN
        assert len(segment_ids) == MAX_LEN

        label_id = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features



def load_and_cache_examples(tokenizer, prefix):
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(DATA_DIR, 'cached_{}_{}_{}'.format(
                                    prefix, "bert", MAX_LEN))

    logger.info("Creating features from dataset file at %s", DATA_DIR)
    processor = MyProcessor()
    examples = processor.get_examples(prefix)

    features = convert_examples_to_features(examples, tokenizer)

    logger.info("Saving features into cached file %s", cached_features_file)
    torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset



#--------------------------------------Evaluate-----------------------------------------------#
from tqdm import tqdm,trange
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import f1_score, classification_report

logger=get_logger(path+"log2")

def evaluate(model, tokenizer, eval_dataset, prefix):

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)

    # Eval
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", BATCH_SIZE)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    epoch_iterator = tqdm(eval_dataloader, desc="Iteration", disable=False)
    for step, batch in enumerate(epoch_iterator):
        model.eval()
        batch = tuple(t.to(DEVICE) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}
            
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)

    acc = 0
    for i in range(len(out_label_ids)):
        if preds[i] == out_label_ids[i]:
            acc += 1
    acc = acc / len(out_label_ids)
    
    
    
    result = classification_report(preds, out_label_ids)
    f1 = f1_score(preds, out_label_ids, average='weighted')

    logger.info("***** Eval results {} *****".format(prefix))
    logger.info("F1 = {}\n".format(f1))
    logger.info(result)
    logger.info("\n===========================\n")

    return eval_loss, f1, result,acc




#--------------------------------------Train-----------------------------------------------#
import math
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler

logger=get_logger(path+"log3")


def train(train_dataset, valid_dataset, model, tokenizer, optimizer_grouped_parameters, stored_dir):

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=ADAM_EPS)
    T_TOTAL = int(len(train_dataloader) * NUM_EPOCH)
    WARMUP_STEP = int(T_TOTAL * WARMUP_PROPORTION)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEP, num_training_steps = T_TOTAL)


    # Training
    logger.info("***** Running training *****")

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.to(DEVICE)
    model.zero_grad()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)# Added here for reproductibility (even between python 2 and 3)

    best_squad_f1 = 0
    best_zalo_f1 = 0
    best_f1 = 0
    best_acc=0

    train_iterator = trange(int(NUM_EPOCH), desc="Epoch", disable=False)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(DEVICE) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]
                      }
            
            ouputs = model(**inputs)
            loss = ouputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            tr_loss += loss.item()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
          
        valid_loss, valid_f1, result,acc = evaluate(model, tokenizer, valid_dataset, "valid")

        
        if acc > best_acc:
            best_acc = acc
            logger.info("======> SAVE BEST MODEL | acc = " + str(acc))
        
        
        if (valid_f1 > best_f1):
            best_f1 = valid_f1
            model.save_pretrained(stored_dir)
            logger.info("======> SAVE BEST MODEL | F1 = " + str(valid_f1))       

    return model, global_step, tr_loss / global_step

import os
from transformers import BertForSequenceClassification
from transformers import BertConfig, BertTokenizer,AutoModel,AutoTokenizer,AutoModelWithLMHead

logger=get_logger(path+"log4")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED) 
TRANSFORMER_MODEL = BertForSequenceClassification


stored_dir = path+"runs/bert_base0"       
if not os.path.exists(stored_dir):
    os.makedirs(stored_dir)


tokenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_pair_large", do_lower_case=True)
config = BertConfig.from_pretrained("clue/roberta_chinese_pair_large", num_labels=LABEL_SIZE)
model = TRANSFORMER_MODEL.from_pretrained("clue/roberta_chinese_pair_large", from_tf=False, config=config)

train_dataset = load_and_cache_examples(tokenizer, TRAIN_DATA) 
valid_dataset = load_and_cache_examples(tokenizer, VALID_DATA) 

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': WEIGHT_DECAY},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


# Model config
logger.info("***** Running training *****")


# Train
model, global_step, tr_loss = train(train_dataset, valid_dataset, model, tokenizer, optimizer_grouped_parameters, stored_dir)
logger.info("global_step = %s, average loss = %s", global_step, tr_loss)
logger.info("\n\n\n============**************===========\n\n\n")



#--------------------------------------Predict TestData-----------------------------------------------#
import random
from tqdm import tqdm
import json


test_dataset = load_and_cache_examples(tokenizer, TEST_DATA)

stored_dir = path+"runs/bert_base0"

config = BertConfig.from_pretrained(stored_dir, num_labels=LABEL_SIZE)
model = TRANSFORMER_MODEL.from_pretrained(stored_dir, from_tf=False, config=config)

model.to(DEVICE)
model.eval()

eval_sampler = SequentialSampler(test_dataset)
eval_dataloader = DataLoader(test_dataset, sampler=eval_sampler, batch_size=1)

epoch_iterator = tqdm(eval_dataloader, desc="Iteration", disable=False)
preds = None

for step, batch in enumerate(epoch_iterator):
    model.eval()
    batch = tuple(t.to(DEVICE) for t in batch)

    with torch.no_grad():
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'labels':         batch[3]}

        outputs = model(**inputs)
        _, logits = outputs[:2]
    if preds is None:
        preds = logits.detach().cpu().numpy()
    else:
        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)



argmax_preds = [np.argmax(p) for p in preds]
df = pd.read_csv("sample_data/test_others.csv", encoding='utf-8')
df["label"] = argmax_preds
df[['id', 'label']].to_csv("sample_data/result_others.csv", index=False)



#--------------------------------------Predict ValidData-----------------------------------------------#
import random
from tqdm import tqdm
import json


dev_dataset = load_and_cache_examples(tokenizer, VALID_DATA)

stored_dir = path+"runs/bert_base0"

config = BertConfig.from_pretrained(stored_dir, num_labels=LABEL_SIZE)
model = TRANSFORMER_MODEL.from_pretrained(stored_dir, from_tf=False, config=config)

model.to(DEVICE)
model.eval()

eval_sampler = SequentialSampler(dev_dataset)
eval_dataloader = DataLoader(dev_dataset, sampler=eval_sampler, batch_size=1)

epoch_iterator = tqdm(eval_dataloader, desc="Iteration", disable=False)
preds = None

for step, batch in enumerate(epoch_iterator):
    model.eval()
    batch = tuple(t.to(DEVICE) for t in batch)

    with torch.no_grad():
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'labels':         batch[3]}

        outputs = model(**inputs)
        _, logits = outputs[:2]
    if preds is None:
        preds = logits.detach().cpu().numpy()
    else:
        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

argmax_preds = [np.argmax(p) for p in preds]
df_dev = pd.read_csv("sample_data/sub_dev_others.csv", encoding='utf-8')
df_dev["label"] = argmax_preds
df_dev[['id', 'label']].to_csv("sample_data/result_dev_others.csv", index=False)



#如果预测的时候colab页面没反应了，重新加载进来，运行这个
import os
from transformers import BertForSequenceClassification
from transformers import BertConfig, BertTokenizer,AutoModel,AutoTokenizer,AutoModelWithLMHead

TRANSFORMER_MODEL = BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_pair_large", do_lower_case=True)

