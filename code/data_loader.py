import re
from tensorflow.keras.utils import to_categorical
from collections import Counter
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np


##############################################################################################################
# Training option
##############################################################################################################
def data_loader_BERT(input_ids, attention_masks, one_hot_labels, batch_size=None, random_state=1234, test_size=0.2, testing=False):
    """generate dataloader for batch input

    Args:
        input_ids ([type]): [description]
        attention_masks ([type]): [description]
        one_hot_labels ([type]): [description]
        batch_size ([type], optional): [description]. Defaults to None.
        random_state (int, optional): [description]. Defaults to 1234.
        test_size (float, optional): [description]. Defaults to 0.2.
        testing (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    assert isinstance(one_hot_labels, np.ndarray)
    stratify_y = np.argmax(one_hot_labels, axis=1)

    one_hot_labels = torch.tensor(one_hot_labels, dtype=torch.int32)
    input_ids = torch.tensor(input_ids, dtype=torch.int32)
    attention_masks = torch.tensor(attention_masks, dtype=torch.int32)

    if not testing:

        train_ids, validation_ids, train_attention_masks, val_attention_masks, train_labels, validation_labels = train_test_split(input_ids, attention_masks, one_hot_labels, random_state=random_state, test_size=test_size, stratify=stratify_y)

        # Create the DataLoader for our training set.  with shuffle
        train_data = TensorDataset(train_ids, train_attention_masks, train_labels)
        train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)

        # Create the DataLoader for our validation set. without shuffle
        validation_data = TensorDataset(validation_ids, val_attention_masks, validation_labels)
        validation_dataloader = DataLoader(validation_data, sampler=SequentialSampler(validation_data), batch_size=batch_size)

        return train_dataloader, validation_dataloader

    else:
        batch_size = len(one_hot_labels) if batch_size is None else batch_size

        test_data = TensorDataset(input_ids, attention_masks, one_hot_labels)
        test_dataloader = DataLoader(test_data, sampler=None, batch_size=batch_size)

        return test_dataloader, None


def k_fold_data_loader_BERT(input_ids, attention_masks, one_hot_labels, batch_size=None, random_state=1234, k_fold=3):
    """generate k-fold dataloader with generator 

    Args:
        input_ids ([type]): [description]
        attention_masks ([type]): [description]
        one_hot_labels ([type]): [description]
        batch_size ([type], optional): [description]. Defaults to None.
        random_state (int, optional): [description]. Defaults to 1234.
        test_size (float, optional): [description]. Defaults to 0.2.
    """
    assert isinstance(one_hot_labels, np.ndarray)
    one_hot_labels = torch.tensor(one_hot_labels, dtype=torch.int32)
    input_ids = torch.tensor(input_ids, dtype=torch.int32)
    attention_masks = torch.tensor(attention_masks, dtype=torch.int32)

    stratify_y = np.argmax(one_hot_labels, axis=1)
    skf = StratifiedKFold(n_splits=k_fold, random_state=random_state, shuffle=True)
    
    for train_index, test_index in skf.split(input_ids, stratify_y):
        X_train_ids, X_test_ids = input_ids[train_index], input_ids[test_index]
        X_train_attention_masks, X_test_attention_masks = attention_masks[train_index], attention_masks[test_index]
        Y_train, Y_test = one_hot_labels[train_index], one_hot_labels[test_index]

        train_data = TensorDataset(X_train_ids, X_train_attention_masks, Y_train)
        train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
        validation_data = TensorDataset(X_test_ids, X_test_attention_masks, Y_test)
        validation_dataloader = DataLoader(validation_data, sampler=SequentialSampler(validation_data), batch_size=batch_size)
        yield train_dataloader, validation_dataloader


def one_hot_encoded_y(labels):
    """
    labels has to be continous
    e.g. wiht 0,1,6, funciton will only output 6 dimension
    """

    import numpy as np
    label_size = max(labels)
    assert label_size > 0
    if len(labels.shape) > 1:
        one_hot_labels = np.eye(label_size)[labels]
    else:
        one_hot_labels = np.eye(2)[labels]
    return one_hot_labels
