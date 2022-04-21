from transformers import BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import ReLU


class bert_lstm_cnn(nn.Module):
    """
    last_hidden_state
    pooler_output
    hidden_states
    attentions
    """

    # define all the layers used in model
    def __init__(self, emb_dim, seq_len, lstm_units, num_filters, kernel_sizes, num_classes, freeze_bert=False, bert_embed_type=1, dropout_rate=0.5):

        super().__init__()

        # self.seq_len = seq_len
        # if you're not using the pooler layer then there's no need to worry about that warning.
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=True, output_hidden_states=True)  # handle sequence length by it self
        self.bert_embed_type = bert_embed_type
        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.emb_dim = emb_dim
        self.lstm_units = lstm_units
        self.lstm = nn.LSTM(self.emb_dim, self.lstm_units, num_layers=1, bidirectional=True, batch_first=True)

        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes  # [1,2,3] -> filter size
        self.num_classes = num_classes
        self.convs = nn.ModuleList([nn.Conv2d(1, self.num_filters, (f, 2 * self.lstm_units)) for f in self.kernel_sizes])
        self.fc = nn.Linear(len(kernel_sizes) * self.num_filters, self.num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)
        

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask) # alread filter by attention mask
        # last_hidden_state
        # pooler_output
        # hidden_states
        # attentions

        if self.bert_embed_type == 3:
            # Extract the last hidden state of the token `[CLS]` for classification task
            x = outputs['last_hidden_state'][:, 0, :]  # [hidden state layer output][batch N, [CLS position], [embedding 768]]
            # batch x 768

        if self.bert_embed_type == 2:
            x = outputs['pooler_output']
            # batch x 768

        if self.bert_embed_type == 1:
            x = torch.stack(outputs['hidden_states'][-4:], dim=0)
            # permute from ]torch.Size([4, 64, 100, 768]) to  torch.Size([64, 100, 4, 768])
            x = x.permute(1, 2, 0, 3)
            # take mean of the last four layers
            x = x.mean(axis=2)
            # (x  * (torch.tile(attention_mask, (768, 1)).T)
            # print(x.shape)
            # batch x seq_len x 768
            # outputs['attentions']
            # @ attention_mask

        x, _ = self.lstm(x)  # (N, seq_len, 2*lstm_units)

        x = x.unsqueeze(1)

        x = [F.relu(conv(x).squeeze(-1)) for conv in self.convs]  # output of three conv

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # continue with 3 maxpooling

        x = torch.cat(x, 1)  # N, len(filter_sizes)* num_filters

        x = self.dropout(x)  # N, len(filter_sizes)* num_filters
        logit = self.fc(x)  # (N, num_classes)

        return logit


class clf_naive(nn.Module):
    # define all the layers used in model
    def __init__(self, emb_dim, seq_len, hidden_units, num_classes, dropout_rate=0):

        super().__init__()
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        # self.cnn_bow_extractor = torch.nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(1, 3),
        # )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.emb_dim, self.emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.emb_dim // 2, self.emb_dim // 4),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.emb_dim // 4, self.hidden_units),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_units, self.num_classes),
        )

    def forward(self, x):
        # depends on input of x, we can extracted from pre trained BERT first to saved memory
        # x = torch.mean(x, axis=1)  # (batchsize, seq_len, emb_dim) -> avg embedding to represent sentence
        # x = x[:, 0,:]  # (batchsize, emb_dim) -> first [CLS] embedding to represent sentence
        logit = self.fc(x)  # (N, num_classes)
        return logit


# Create the BertClassfier class
class clf_finetuneBERT(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, emb_dim=768, freeze_bert=False, num_classes=20, hidden_units=50, bert_embed_type=3):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(clf_finetuneBERT, self).__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.bert_embed_type = bert_embed_type
        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_units = hidden_units

        self.classifier = nn.Sequential(
            nn.Linear(self.emb_dim, self.hidden_units),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(self.hidden_units, num_classes))

        # Freeze the BERT model
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

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.bert_embed_type == 3:
            # Extract the last hidden state of the token `[CLS]` for classification task
            x = outputs[0][:, 0, :]  # [hidden state layer output][batch N, [CLS position], [embedding 768]]
            # batch x 768

        if self.bert_embed_type == 2:
            x = outputs.pooler_output
            # batch x 768

        if self.bert_embed_type == 1:
            x = torch.stack(outputs[2][-4:], dim=0).permute(1, 2, 0, 3).mean(axis=2)
            # batch x seq_len x 768

        logits = self.classifier(x)
        return logits




class clf(nn.Module):

    # define all the layers used in model
    def __init__(self, emb_dim, seq_len, lstm_units, num_classes, dropout_rate=0.2):

        super().__init__()
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.lstm_units = lstm_units
        self.num_classes = num_classes

        # self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_index)
        self.lstm = nn.LSTM(
            emb_dim,
            lstm_units,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,  # for multiple layers
            batch_first=True)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(2 * lstm_units, self.num_classes),
        )

    def forward(self, x):
        packed_output, (h_T, c_T) = self.lstm(x)  # (N, seq_len, 2*lstm_units)
        # h_T = [N, num layers * num directions, hid dim] => 最后一个 timestamp 输出的 vector
        # c_T = [N, num layers * num directions, hid dim] => 最后一个 timestamp 输出的 vector
        # packed_output = N, seq_len, num_directions * hidden_size
        # print(packed_output[:,-1,:].size())
        hidden = torch.cat((h_T[-2, :, :], h_T[-1, :, :]), dim=1)  # 取最后两层, 测试是否 work # 80% acc
        # hidden = torch.cat((c_T[-2,:,:], c_T[-1,:,:]), dim = 1) # 取最后两层, 测试是否 work # 81% acc
        # hidden = packed_output[:,-1,:] # [4, 16, 160]
        # hidden = packed_output[:,-1,:] # [4, 16, 160]
        # print(hidden.size())

        logit = self.fc(hidden)  # (N, num_classes)
        # prop = torch.sigmoid(logit)  # Sigmoid for multilabel prediction + BCEWithLogitsLoss
        # print(prop.shape)
        return logit


class lstm_cnn(nn.Module):
    '''The output dimension is one, use the torch.nn.BCELoss(reduction='mean')'''

    # define all the layers used in model
    def __init__(self, emb_dim, seq_len, lstm_units, num_filters, kernel_sizes, num_classes, dropout_rate=0.5):
        super().__init__()
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.lstm_units = lstm_units
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.num_classes = num_classes
        self.lstm = nn.LSTM(emb_dim, lstm_units, num_layers=1, bidirectional=True, batch_first=True)
        self.convs = nn.ModuleList([nn.Conv2d(1, self.num_filters, (f, 2 * self.lstm_units)) for f in self.kernel_sizes])
        self.dropout = nn.Dropout(p=dropout_rate)
        # self.fc = nn.Sequential(nn.Linear(len(kernel_sizes) * self.num_filters, 1), nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(len(kernel_sizes) * self.num_filters, self.num_classes))

    def forward(self, x):
        x, _ = self.lstm(x)  # (N, seq_len, 2*lstm_units)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x).squeeze(-1)) for conv in self.convs]  # output of three conv
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # continue with 3 maxpooling
        x = torch.cat(x, 1)  # N, len(filter_sizes)* num_filters
        x = self.dropout(x)  # N, len(filter_sizes)* num_filters
        logit = self.fc(x)  # (N, 1)

        return logit


class bert_dense(nn.Module):
    '''The output dimension is two, use the BCE loss with logit'''

    # define all the layers used in model
    def __init__(self, kernel_sizes, num_classes, freeze_bert=False, bert_embed_type=1, dropout_rate=0.5):

        super().__init__()
        self.bert_embed_type = bert_embed_type
        self.num_classes = num_classes
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(len(kernel_sizes) * self.num_filters, self.num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.bert_embed_type == 3:
            # Extract the last hidden state of the token `[CLS]` for classification task
            x = outputs[0][:, 0, :]  # [hidden state layer output][batch N, [CLS position], [embedding 768]]
            # batch x 768

        if self.bert_embed_type == 2:
            x = outputs.pooler_output
            # batch x 768

        logit = self.fc(x)  # (N, num_classes)

        return logit
