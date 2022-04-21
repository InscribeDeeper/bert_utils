### This file prepared is prepared in my previously projects.
import re
import random
import matplotlib.pyplot as plt
import datetime
import torch
import numpy as np
import sys
from torch.nn.functional import softmax, sigmoid
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score  # average_precision_score
import time
import copy
import os
from tqdm import tqdm

try:
    import glovar
except ImportError:
    print("Append path: ", os.path.abspath(os.path.dirname(__file__)))
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    import glovar


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return


def text_to_tokens(sentences, tokenizer, max_len=100):

    input_ids = []
    attention_masks = []

    for sent in tqdm(sentences):
        encoded_dict = tokenizer.encode_plus(sent, truncation=True, add_special_tokens=True, max_length=max_len, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def select_bert_emb(outputs, embed_type):
    """
        ## in output
        # 0.last_hidden_state
        # 1.pooler_output
        # 2.hidden_states
        # 3.attentions

        ## dimension
        # {'attentions': 12 layer * torch.Size([1, 12, 100, 100]),
        # 'hidden_states': (12 layer + 1 output layer) *  torch.Size([1, 100, 768]),
        # 'last_hidden_state': torch.Size([100, 768]),
        # 'pooler_output': torch.Size([768])}
    """

    if embed_type == 1:  # to get contextual embedding for each words
        x = torch.stack(outputs['hidden_states'][-4:], dim=0)
        # permute from ]torch.Size([4, 64, 100, 768]) to  torch.Size([64, 100, 4, 768])
        x = x.permute(1, 2, 0, 3)
        # take mean of the last four layers
        x = x.mean(axis=2)
        # (x  * (torch.tile(attention_mask, (768, 1)).T)
    elif embed_type == 2:
        x = outputs['pooler_output']
    elif embed_type == 3:  # only finetune version will have a good result based on [CLS]
        x = outputs['last_hidden_state'][:, 0, :]  # [hidden state layer output][batch N, [CLS position], [embedding 768]]
    elif embed_type == 4:  # only finetune version will have a good result based on [CLS]
        x = outputs['last_hidden_state']  # [hidden state layer output][batch N, [CLS position], [embedding 768]]
        x = x.mean(axis=1)
    return x


def get_metrics(true_labels, pred_labels, num_labels):
    pred_sparse = np.where(pred_labels > 0.5, 1, 0) if num_labels == 1 else np.argmax(pred_labels, axis=1)
    true_sparse = np.where(true_labels > 0.5, 1, 0) if num_labels == 1 else np.argmax(true_labels, axis=1)
    ## metrics
    auc_score = roc_auc_score(true_labels, pred_labels, multi_class='ovr')
    precison = precision_score(true_sparse, pred_sparse, average='macro')
    recall = recall_score(true_sparse, pred_sparse, average='macro')
    acc = accuracy_score(true_sparse, pred_sparse)
    f1 = f1_score(true_sparse, pred_sparse, average='macro')
    return auc_score, precison, recall, acc, f1


def text_to_bert_embedding(sentences, bert_name='bert-base-uncased', max_len=100, device=glovar.device_type, low_RAM_mode=False, embed_type=2):
    """Generate BERT embedding 

    Args:
        sentences ([list, pd.Series]): [list of sentences]
        tokenizer ([type]): [predefined BERT tokenizer]
        max_len (int, optional): [description]. Defaults to 100.
        device ([type], optional): [description]. Defaults to glovar.device_type.
        low_RAM_mode (bool, optional): [For low V-RAM option]. Defaults to False.
        embed_type (int, optional): [
                                    # 0.last_hidden_state
                                    # 1.pooler_output
                                    # 2.hidden_states
                                    # 3.attentions
                                    # ]. Defaults to 2.

    Returns:
        [np.array]: [input_ids, output_embedding, attention_masks]
    """

    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    bert_model = BertModel.from_pretrained(bert_name, output_attentions=True, output_hidden_states=True).to(device)  # handle sequence length by it self

    input_ids = []
    attention_masks = []
    output_embedding = []

    for sent in tqdm(sentences):
        encoded_dict = tokenizer.encode_plus(sent, truncation=True, add_special_tokens=True, max_length=max_len, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    if low_RAM_mode:
        with torch.no_grad():
            # too large -> for loop to V-RAM handle
            for batch_input_ids, batch_attention_masks in tqdm(list(zip(input_ids, attention_masks))):
                outputs = bert_model(batch_input_ids.to(device), batch_attention_masks.to(device))
                tokens_embeddings = select_bert_emb(outputs, embed_type)
                output_embedding.append(tokens_embeddings.squeeze().cpu().numpy())  # transfer into cpu to save V-RAM

            output_embedding = np.array(output_embedding)  # np.array([np.array([2,4]), np.array([2,4]), np.array([2,4])]) -> shape (3,2)
            input_ids = torch.cat(input_ids, dim=0).cpu().numpy()  # equal to np.vstack
            attention_masks = torch.cat(attention_masks, dim=0).cpu().numpy()

    else:  # high V-RAM outer batch
        with torch.no_grad():
            input_ids = torch.cat(input_ids, dim=0).to(device)  # equal to np.vstack
            attention_masks = torch.cat(attention_masks, dim=0).to(device)
            outputs = bert_model(input_ids, attention_masks)
            tokens_embeddings = select_bert_emb(outputs, embed_type)
            output_embedding = tokens_embeddings.cpu().numpy()  # transfer into cpu to save V-RAM

    return input_ids, output_embedding, attention_masks


def select_loss_function(num_labels, device='cpu', class_weight=None):
    pos_weight = torch.tensor(class_weight).to(device) if class_weight is not None else None
    if num_labels > 1:
        criterion = CrossEntropyLoss()
    elif num_labels == 1:
        criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
    return criterion


def pred_bert_model(model, dataloader, num_labels, class_weight=None, task='Train', verbose=2):
    '''only move the data into GPU when training and validating'''

    device = glovar.device_type
    model.eval()
    tokenized_texts = []
    logit_preds = []
    b_true_labels = []
    b_pred_labels = []
    # total_eval_accuracy = 0
    total_eval_loss = 0
    # nb_eval_steps = 0

    criterion = select_loss_function(num_labels, device, class_weight)

    for batch in dataloader:
        with torch.no_grad():
            b_input_ids = batch[0].to(device)  # [0]: input sentence ids
            b_attentions = batch[1].to(device)  # [1]: att
            b_labels = batch[2].to(device)  # [2]: labels
            b_logits = model(b_input_ids, b_attentions)

            if num_labels > 1:
                b_prob = softmax(b_logits, dim=1)
                val_loss = criterion(b_prob, b_labels.type_as(b_prob))  # convert labels to float for calculation
            elif num_labels == 1:
                val_loss = criterion(b_logits, b_labels.type_as(b_prob))  # convert labels to float for calculation

            total_eval_loss += val_loss.item()

            # # save result
            true_label = b_labels.detach().cpu().numpy()
            pred_label = b_prob.detach().cpu().numpy()

            logit_preds.append(b_logits)
            b_true_labels.append(true_label)
            b_pred_labels.append(pred_label)
            tokenized_texts.append(b_input_ids)

    # Flatten outputs
    pred_labels = np.vstack(b_pred_labels)
    true_labels = np.vstack(b_true_labels)
    avg_val_loss = total_eval_loss / len(dataloader)

    auc_score, precison, recall, val_acc, f1 = get_metrics(true_labels, pred_labels, num_labels)

    if verbose > 1:
        print(f"      {task} Loss: {round(avg_val_loss,4)}\t {task} Acc: {round(val_acc,4)}\t {task} F1: {round(f1,4)}\t{task} ovr AUC: {round(auc_score,4)}")

    return tokenized_texts, pred_labels, true_labels, avg_val_loss, val_acc


def fit_bert_model(model, num_labels, train_dataloader, validation_dataloader, optimizer=None, scheduler=None, epochs=10, class_weight=None, patience=3, model_path='bert_clf.pt', verbose=0):
    """
    Below is our training loop. There's a lot going on, but fundamentally for each pass in our loop we have a trianing phase and a validation phase. At each pass we need to:

    Training loop:
    - Unpack our data inputs and labels
    - Load data onto the GPU for acceleration
    - Clear out the gradients calculated in the previous pass.
        - In pytorch the gradients accumulate by default (useful for things like RNNs) unless you explicitly clear them out.
    - Forward pass (feed input data through the network)
    - Backward pass (backpropagation)
    - Tell the network to update parameters with optimizer.step()
    - Track variables for monitoring progress

    Evalution loop:
    - Unpack our data inputs and labels
    - Load data onto the GPU for acceleration
    - Forward pass (feed input data through the network)
    - Compute loss on our validation data and track variables for monitoring progress
    The loss function is different from multi-label classifer

    Parameters:

    * model: model defined
    *   num_labels: number of labels
    *   train_dataloader: train data loader
    *   validation_dataloader: validation data loader
    *   optimizer: optimizer. default is Adam
    *   scheduler: adjust learning rate dynamically; default is None.
    *   epochs: number of epochs
    """

    device = glovar.device_type
    print(device)
    criterion = select_loss_function(num_labels, device, class_weight)

    training_stats = []
    best_score = -999
    best_epoch = 0
    best_model = copy.deepcopy(model.state_dict())

    total_t0 = time.time()

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    if not os.path.exists(model_path[0:model_path.rfind("/")]):
        os.makedirs(model_path[0:model_path.rfind("/")])

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================
        print("")
        print(f'======== Epoch {epoch_i + 1} / {epochs} ========')
        t0 = time.time()
        epoch_train_loss = 0
        epoch_train_acc = 0
        model.train()
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                if verbose > 1:
                    print(f'  Batch {step}  of  {len(train_dataloader)}.    Elapsed: {elapsed}.')

            model.zero_grad()  # zero out the accumulated gradient before back prop
            b_input_ids = batch[0].to(device)  # [0]: input sentence ids
            b_attentions = batch[1].to(device)  # [1]: att
            b_labels = batch[2].to(device)  # [2]: labels
            b_logits = model(b_input_ids, b_attentions)

            if num_labels > 1:
                b_prob = softmax(b_logits, dim=1)
                loss = criterion(b_prob, b_labels.type_as(b_prob))  # convert labels to float for calculation
            elif num_labels == 1:
                b_prob = sigmoid(b_logits)
                loss = criterion(b_logits, b_labels.type_as(b_logits))

            epoch_train_loss += loss.item()
            pred_bools = np.argmax(b_prob.detach().cpu().numpy(), axis=1)
            true_bools = np.argmax(b_labels.detach().cpu().numpy(), axis=1)
            epoch_train_acc += (pred_bools == true_bools).astype(int).sum() / len(pred_bools)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 可以试着删除

            clip_value = False
            if clip_value:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        training_time = format_time(time.time() - t0)
        print(f"  Training epcoh took: {training_time}")

        # ========================================
        #               Collection train and validation information
        # ========================================
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        avg_train_acc = epoch_train_acc / len(train_dataloader)
        print(f"      Train Loss: {round(avg_train_loss,4)}\t Train Acc: {round(avg_train_acc,4)}\t", end='')

        t0 = time.time()
        tokenized_texts, pred_labels, val_labels, val_loss, val_acc = pred_bert_model(model, validation_dataloader, num_labels, class_weight=class_weight, task='Val')
        training_stats.append({
            'epoch': epoch_i + 1,
            'train_loss': avg_train_loss,
            'train_acc': avg_train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'Best epoch': best_epoch,
            'Training Time': training_time,
            'Validation Time': format_time(time.time() - t0),
        })

        # ========================================
        #               Early stop
        # ========================================
        auc_score, _, _, _, _ = get_metrics(val_labels, pred_labels, num_labels)
        if auc_score > best_score:
            best_score = auc_score
            best_epoch = epoch_i + 1
            best_model = copy.deepcopy(model.state_dict())
            print("             best_model updated based on ovr AUC !")
            cnt = 0
        else:
            cnt += 1
            if cnt == patience:
                print("\n")
                print("early stopping at epoch {0}".format(epoch_i + 1))
                break

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
    torch.save(best_model, model_path)
    print(f"best_model saved in {model_path}")
    return model, training_stats


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def device_checking(cpu_pref=False):
    # CUDA_LAUNCH_BLOCKING = 1

    # If there's a GPU available...
    if torch.cuda.is_available() and (not cpu_pref):

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def re_softmax(y, axis=1):
    """Compute softmax values for each sets of scores in x."""
    x = np.log(y / (1 - y))  # inverse of sigmoid to logits
    if axis == 0:
        return np.exp(x) / np.sum(np.exp(x), axis=0)  # logits to softmax
    else:
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)


def plot_bert_history(df_stats, hold=False, comb=[['train_acc', 'val_acc'], ['train_loss', 'val_loss']]):
    colors = ['navy', 'darkorange'] if hold is False else [None, None] 
    for train_metric, val_metric in comb:
        print("========================================================================")
        xx = list(range(1, len(df_stats) + 1))
        plt.plot(xx, df_stats[train_metric], color=colors[0], lw=2, label=train_metric)
        plt.plot(xx, df_stats[val_metric], color=colors[1], lw=2, label=val_metric)
        plt.title(f"{train_metric} v.s. {val_metric}")
        name = re.split(" |_", train_metric)[-1]
        plt.xlabel('Epochs')
        plt.ylabel(name)
        plt.title(f"Finetune BERT training {name}")
        plt.legend()
        if not hold:
            plt.show()
