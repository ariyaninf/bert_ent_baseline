import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
nltk.download('punkt')
from transformers import BertModel
from pytorch_metric_learning import losses, miners

class BertModel_FineTuned(torch.nn.Module):
    def __init__(self, config, bert_version='bert-base-uncased'):
        super(BertModel_FineTuned, self).__init__()
        self.bert = BertModel.from_pretrained('libs/bert_model_base_uncased', output_hidden_states=True, local_files_only=True)
        self.in_dim = self.bert.config.hidden_size # in_dim = 768
        self.pooling = config.pooling
        self.embedding = config.embedding
        self.hid_dim = 384  # hidden_dim = 768 / 2
        self.hid_dim_hadamard_cat = 1152  # hidden_dim = 768 * 3/2
        self.trans1 = nn.Linear(self.in_dim, self.hid_dim, bias=True)
        self.trans1_concat = nn.Linear(self.in_dim * 2, self.in_dim, bias=True)
        self.trans1_hadamard_cat = nn.Linear(self.in_dim * 3, self.hid_dim_hadamard_cat, bias=True)
        self.layer_out = nn.Linear(self.hid_dim, 1, bias=True)
        self.layer_out_concat = nn.Linear(self.in_dim, 1, bias=True)
        self.layer_out_hadamard_cat = nn.Linear(self.hid_dim_hadamard_cat, 1, bias=True)
        self.relu = nn.ReLU()
        self.batch_size = config.batch_size
        self.register_buffer('model', None)

    def forward(self, token_ids1, input_mask1, token_ids2, input_mask2, labels):
        emb_in1 = self.get_bert_emb(token_ids1, input_mask1, self.pooling)
        emb_in2 = self.get_bert_emb(token_ids2, input_mask2, self.pooling)

        if torch.cuda.is_available():
            labels = torch.tensor(labels).to("cuda")

        loss = self.en_loss(emb_in1, emb_in2, labels)
        return loss

    def get_bert_emb(self, token_ids, input_masks, pooling):
        # get the embedding from the last layer
        outputs = self.bert(input_ids=token_ids, attention_mask=input_masks)
        last_hidden_states = outputs.hidden_states[-1]

        if pooling is None:
            pooling = 'cls'
        if pooling == 'cls':
            pooled = last_hidden_states[:, 0, :]
        elif pooling == 'mean':
            pooled = last_hidden_states.sum(axis=1) / input_masks.sum(axis=-1).unsqueeze(-1)
        elif pooling == 'max':
            pooled = torch.max((last_hidden_states * input_masks.unsqueeze(-1)), axis=1)
            pooled = pooled.values
        del token_ids
        del input_masks
        return pooled

    def en_loss(self, trans_in1, trans_in2, en_labels):
        batch_size = trans_in1.shape[0]
        z_in1 = F.normalize(trans_in1, dim=1)
        z_in2 = F.normalize(trans_in2, dim=1)
        t_labels = en_labels

        pred_all = torch.tensor([]).to(trans_in1.device)

        for i in range(0, batch_size):
            emb_in_1 = z_in1[i]
            emb_in_2 = z_in2[i]
            if self.embedding == 'concat':
                emb_in = torch.cat([emb_in_1, emb_in_2], dim=0)  # tensor dimension size [2 * 768]
                h1 = self.relu(self.trans1_concat(emb_in))
                x = self.layer_out_concat(h1)
            elif self.embedding == 'subtract':
                emb_in = torch.sub(emb_in_2 - emb_in_1)  # tensor dimension size [768]
                h1 = self.relu(self.trans1(emb_in))
                x = self.layer_out(h1)
            elif self.embedding == 'hadamard_cat':
                emb_in = torch.mul(emb_in_1, emb_in_2)  # hadamard product vector1 and vector2
                emb_in = torch.cat([emb_in, emb_in_1], dim=0)  # concatenates hadamard product with vector1
                emb_in = torch.cat([emb_in, emb_in_2], dim=0)  # concatenate emb_in with vector2, tensor dimension size [3 * 768]
                h1 = self.relu(self.trans1_hadamard_cat(emb_in))
                x = self.layer_out_hadamard_cat(h1)

            # collect all predictions
            pred_all = torch.cat([pred_all, x], dim=0)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(pred_all, t_labels)

        del t_labels, pred_all
        return loss


def tokenize_mask_clauses(clauses, max_seq_len, tokenizer):
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    tokens = [cls_token]

    tokens += tokenizer.tokenize(clauses)
    tokens += [sep_token]

    # if tokens is longer than max_seq_len
    if len(tokens) >= max_seq_len:
        tokens = tokens[:max_seq_len - 2] + [sep_token]
    assert len(tokens) <= max_seq_len

    t_id = tokenizer.convert_tokens_to_ids(tokens)
    # add padding
    padding = [0] * (max_seq_len - len(t_id))
    i_mask = [1] * len(t_id) + padding
    t_id += padding
    return t_id, i_mask


def tokenize_mask(sentences, max_seq_len, tokenizer):
    token_ids = []
    input_mask = []
    for sent in sentences:
        t_id, i_mask = tokenize_mask_clauses(sent, max_seq_len, tokenizer)
        token_ids.append(t_id)
        input_mask.append(i_mask)
    return token_ids, input_mask


def convert_tuple_to_tensor(input_tuple, use_gpu=False):
    token_ids, input_masks = input_tuple
    token_ids = torch.tensor(token_ids, dtype=torch.long)
    input_masks = torch.tensor(input_masks, dtype=torch.long)
    if use_gpu:
        token_ids = token_ids.to("cuda")
        input_masks = input_masks.to("cuda")
    return token_ids, input_masks


class EarlyStopping:
    def __init__(self, patience=3, verbose=True, delta=1e-7, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # reset counter

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


