# Bert 处理代码框架

## RoBERTa
- Bert 是 undertrain
- 更多的epochs和数据

## XLNet
- Relative position embedding 是在这个BERTXL 上最大的改进
    - Relative attention
        - 看self-attention依赖的一个词, 同时用非线性结构 把其他词的 attention 也组合起来
    - Absolute attention
        - 只看self-attention依赖的一个词
- Permutation LM
- train with 更多数据


## K-fold
- 能够将validation set 也用来 train 不会损失数据
- 能够得到稳定的平均值, 确定 reducable
## ALBERT
- share embedding accross all layers
- embedding projection matrix => decomposition
- 只是参数更小了, 但是train的速度没有提高

## T5
- 

## ELECTRA
- 

## BERT size

-   colab - instance
-   bert large
    -   335M parameters, 1024 embed size, 24 layers, 1.3G
-   bert base
    -   768 embed size, 12 layers, 110M params, 400 mb

## Distill Version
- 

## Detail

-   BERT for QA
    -   在最后一层 都会 multiply [CLS] 并且经过 softmax, 作为 start attention,
    -   在最后一层 都会 multiply [SEP] 并且经过 softmax, 作为 end attention
    -   whole word masking -> 不会把词拆分
-   seq-len 大于 200 bert 的 global pooler output 不 work 了 如果大部分词汇都是空的
-   padding

    -   attention maks 是用来 ignore 哪些 tokens 的

-   LM
    -   只考虑 left or right context
    -   不像 bidirectional, word 可以 see 所有的词, transformer
    -

## 精简 example

```python

# !pip install transformers
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions = True, output_hidden_states = True)

sent ='NLP is nice this is the length test for wyang. '
max_len = 10
inputs = tokenizer.encode_plus(sent, truncation=True, add_special_tokens=True, max_length=max_len, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')

outputs = model(**inputs)


# in output
0.last_hidden_state
1.pooler_output
2.hidden_states
3.attentions

# dimension
{'attentions': 12 layer * torch.Size([1, 12, 100, 100]),
'hidden_states': (12 layer + 1 output layer) *  torch.Size([1, 100, 768]),
'last_hidden_state': torch.Size([100, 768]),
'pooler_output': torch.Size([768])}





# b = {}
# for k in inputs.keys():
#     for a in inputs[k]:
#         b[k] = a.shape
# print(inputs, '\n')
# print(b,'\n')
# print(tokenizer.decode(inputs.input_ids[0]), '\n')

# for k in outputs.keys():
#     print(k)
#     for i, a in enumerate(outputs[k]):
#         b[k] = b.get(k, []) + [("L-" + str(i+1), a.shape)]

# b

{'input_ids': tensor([[  101, 17953,  2361,  2003,  3835,  2023,  2003,  1996,  3091,   102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}

{'input_ids': torch.Size([10]), 'token_type_ids': torch.Size([10]), 'attention_mask': torch.Size([10])}

[CLS] nlp is nice this is the length [SEP]


{'attention_mask': torch.Size([10]),
 'attentions': [('L-1', torch.Size([1, 12, 10, 10])),
  ('L-2', torch.Size([1, 12, 10, 10])),
  ('L-3', torch.Size([1, 12, 10, 10])),
  ('L-4', torch.Size([1, 12, 10, 10])),
  ('L-5', torch.Size([1, 12, 10, 10])),
  ('L-6', torch.Size([1, 12, 10, 10])),
  ('L-7', torch.Size([1, 12, 10, 10])),
  ('L-8', torch.Size([1, 12, 10, 10])),
  ('L-9', torch.Size([1, 12, 10, 10])),
  ('L-10', torch.Size([1, 12, 10, 10])),
  ('L-11', torch.Size([1, 12, 10, 10])),
  ('L-12', torch.Size([1, 12, 10, 10]))],
 'hidden_states': [('L-1', torch.Size([1, 10, 768])),
  ('L-2', torch.Size([1, 10, 768])),
  ('L-3', torch.Size([1, 10, 768])),
  ('L-4', torch.Size([1, 10, 768])),
  ('L-5', torch.Size([1, 10, 768])),
  ('L-6', torch.Size([1, 10, 768])),
  ('L-7', torch.Size([1, 10, 768])),
  ('L-8', torch.Size([1, 10, 768])),
  ('L-9', torch.Size([1, 10, 768])),
  ('L-10', torch.Size([1, 10, 768])),
  ('L-11', torch.Size([1, 10, 768])),
  ('L-12', torch.Size([1, 10, 768])),
  ('L-13', torch.Size([1, 10, 768]))],
 'input_ids': torch.Size([10]),
 'last_hidden_state': [('L-1', torch.Size([10, 768]))],
 'pooler_output': [('L-1', torch.Size([768]))],
 'token_type_ids': torch.Size([10])}





##################################################################

sent = 'IP is nice. '

{'input_ids': tensor([[  101, 12997,  2003,  3835,  1012,   102,     0,     0,     0,     0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])}

{'input_ids': torch.Size([10]), 'token_type_ids': torch.Size([10]), 'attention_mask': torch.Size([10])}

[CLS] ip is nice. [SEP] [PAD] [PAD] [PAD] [PAD]

{'attention_mask': torch.Size([10]),
 'attentions': torch.Size([1, 12, 10, 10]),
 'hidden_states': 12 layer * torch.Size([1, 10, 768]),
 'input_ids': torch.Size([10]),
 'last_hidden_state': torch.Size([10, 768]),
 'pooler_output': torch.Size([768]),
 'token_type_ids': torch.Size([10])}



##################################################################
#### 特殊词汇被拆成两个 17953,  2361 = NLP
#### 这里长度为10, [CLS] nlp is nice. [SEP] [PAD] [PAD] [PAD], 其中 NLP 算2个词, 17953,  2361 = NLP
##################################################################
sent = 'NLP is nice. '

{'input_ids': tensor([[  101, 17953,  2361,  2003,  3835,  1012,   102,     0,     0,     0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])}

{'input_ids': torch.Size([10]), 'token_type_ids': torch.Size([10]), 'attention_mask': torch.Size([10])}

[CLS] nlp is nice. [SEP] [PAD] [PAD] [PAD]

{'attention_mask': torch.Size([10]),
 'attentions': torch.Size([1, 12, 10, 10]),
 'hidden_states': 12 layer * torch.Size([1, 10, 768]),
 'input_ids': torch.Size([10]),
 'last_hidden_state': torch.Size([10, 768]),
 'pooler_output': torch.Size([768]),
 'token_type_ids': torch.Size([10])}



##################################################################
#### 特殊词汇被拆成两个 17953,  2361 = NLP
#### 这里长度为10, 因为NLP算两个, 所以最后保存下来只有 7个词, 整个 ids中包括8个 ids, 然后前后一个是CLS,一个是SEP
#### 也就是说 max_len 包括了 [CLS] 和 [SEP], 而且特殊词汇被拆分 ## 主要是 add_special_tokens 这个参数, 会返回[CLS] 和 [SEP]

#### attention 每一层都会有, 10 x 10 是自己与自己的 pair attention score
#### hidden state 是最后一层的 每一个词 的 hidden state
#### pooler_output 是最后一层的 hidden state global 处理集中过的一种 代表句子的 embedding
#### attention mask 要保留下来 与 抽出来的embedding 相乘
##################################################################

sent = 'NLP is nice this is the length test for wyang '

{'input_ids': tensor([[  101, 17953,  2361,  2003,  3835,  2023,  2003,  1996,  3091,   102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}

{'input_ids': torch.Size([10]), 'token_type_ids': torch.Size([10]), 'attention_mask': torch.Size([10])}

[CLS] nlp is nice this is the length [SEP]

{'attention_mask': torch.Size([10]),
 'attentions': torch.Size([1, 12, 10, 10]),
 'hidden_states': 12 layer * torch.Size([1, 10, 768]),
 'input_ids': torch.Size([10]),
 'last_hidden_state': torch.Size([10, 768]),
 'pooler_output': torch.Size([768]),
 'token_type_ids': torch.Size([10])}





##################################################################
#### tokenizer.plus 每次只返回一个
##################################################################


sent = ['NLP is nice this is the length test for wyang ', 'I am good. ']



{'input_ids': tensor([[101, 100, 100, 102,   0,   0,   0,   0,   0,   0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]])}

{'input_ids': torch.Size([10]), 'token_type_ids': torch.Size([10]), 'attention_mask': torch.Size([10])}

[CLS] [UNK] [UNK] [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]

{'attention_mask': torch.Size([10]),
 'attentions': torch.Size([1, 12, 10, 10]),
 'hidden_states': 12 layer * torch.Size([1, 10, 768]),
 'input_ids': torch.Size([10]),
 'last_hidden_state': torch.Size([10, 768]),
 'pooler_output': torch.Size([768]),
 'token_type_ids': torch.Size([10])}


```

## tokenizer

sentence = 'I love Beijing'

tokens = tokenizer.tokenize(sentence)
print(tokens)

tokens = ['[CLS]'] + tokens + ['[SEP]']
print(tokens)

tokens = tokens + ['[PAD]'] \* 2
print(tokens)

['i', 'love', 'beijing']
['[CLS]', 'i', 'love', 'beijing', '[SEP]']
['[CLS]', 'i', 'love', 'beijing', '[SEP]', '[PAD]', '[PAD]']

### mask

attention_mask = [ 1 if t != '[PAD]' else 0 for t in tokens]
print(attention_mask)

[1, 1, 1, 1, 1, 0, 0]

### token id

token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)

[101, 1045, 2293, 7211, 102, 0, 0]

### To tensor

token_ids = tf.convert_to_tensor(token_ids)
token_ids = tf.reshape(token_ids, [1, -1])

attention_mask = tf.convert_to_tensor(attention_mask)
attention_mask = tf.reshape(attention_mask, [1, -1])

## Output

output = model(token_ids, attention_mask = attention_mask)
print(output[0].shape, output[1].shape)
(1, 7, 768) (1, 768)

BERT 模型最后一层的输出。由于输入有 7 个 tokens，所以对应有 7 个 token 的 Embedding。其对应的维度为（batch_size, sequence_length, hidden_size）
输出层中第 1 个 token（这里也就是对应 的[CLS]）的 Embedding，并且已被一个线性层 + Tanh 激活层处理。线性层的权重由 NSP 作业预训练中得到。其对应的维度为（batch_size, hidden_size）

BERT 的研究人员做了进一步研究。在命名体识别任务中，研究人员除了使用 BERT 最后一层的输出作为提取的特征外，还尝试使用了其他层的输出（例如通过拼接的方式进行组合. 在使用最后 4 层（h9 到 h12 层的拼接）的输出时，能得到比仅使用 h12 的输出更高的 F1 分数 96.1。

## 其他 load pretrained bert model

from transformers import TFBertForSequenceClassification, BertTokenizerFast

```python
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# tokenize every sequence
def bert_encoder(review):
    encoded = tokenizer(review.numpy().decode('utf-8'), truncation=True, max_length=150, pad_to_max_length=True)
    return encoded['input_ids'], encoded['token_type_ids'], encoded['attention_mask']

bert_train = [bert_encoder(r) for r, l in imdb_train]
bert_label = [l for r, l in imdb_train]

bert_train = np.array(bert_train)
bert_label = tf.keras.utils.to_categorical(bert_label, num_classes=2)

print(bert_train.shape, bert_label.shape)
(25000, 3, 150) (25000, 2)

```

这里需要注意的是，我们使用的是 TFBertForSequenceClassification 和 BertTokenizerFast。TFBertForSequenceClassification 是包装好的类，专门用于做分类，由 1 层 bert、1 层 Dropout、1 层前馈网络组成，其定义可以参考官网[5]。BertTokenizerFast 也是一个方便的 tokenizer 类，会比 BertTokenizer 更快一些。

## reference:

-   BERT Keras Version: https://www.cnblogs.com/zackstang/p/15387549.html
-   NLP 任务: https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/

<!-- https://www.cnblogs.com/zackstang/p/8232921.html -->

## Packages:

-   scikit-learn：python 里的机器学习库
-   Natural Language Toolkit（NLTK）：包含所有 NLP 技术的完整工具
-   Pattern：一个 web mining 模块，用于 NLP 和机器学习
-   TextBlob：操作简单的 nlp 工具 API，构建于 NLTK 和 Pattern
-   spaCy：Industrial strength NLP with Python and Cython
-   Gensim：主题建模
-   Stanford Core NLP：Stanford NLP group 提供的 NLP 服务包

### notes

-   如果我不需要每一个词的 contextual embedding, 那我就只需要单独的提取均值, 或者[CLS]的 embedding = 第 0 个 embed 就行
