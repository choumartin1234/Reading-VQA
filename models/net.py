import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

from .fusion import MutanFusion

class Net(nn.Module):
    def __init__(self, device, dim_bert=768, dim_v=2048, dim_q=2048, dim_c=1024, dim_r=512, dim_o=300):
        super().__init__()
        self.device = device
        self.dim_bert = dim_bert
        self.dim_c = dim_c
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout_v = nn.Dropout(0.5)
        self.dropout_q = nn.Dropout(0.3)
        self.linear_q = nn.Linear(dim_bert, dim_q)
        self.dropout_c = nn.Dropout(0.3)
        self.lstm_c = nn.LSTM(dim_bert, dim_c, bidirectional=True)

        self.fusion_v_c = MutanFusion(dim_v, 768, dim_c, 512, dim_r)
        self.fusion_r_q = MutanFusion(dim_r, dim_o, dim_q, dim_o, dim_o)
        self.fc = nn.Linear(dim_o, self.tokenizer.vocab_size)

    def forward(self, data):
        with torch.no_grad():
            data['answer'] = torch.tensor([x[0] for x in self.tokenizer.batch_encode_plus(data['answer'])['input_ids']]).to(self.device)

        x_v = self.dropout_v(data['image'].mean(3).mean(2).to(self.device))
        x_q = self.get_question(data)
        x_c = self.get_contexts(data)
        # x_v = self.attention(x_v, x_q)
        x_r = torch.softmax(self.fusion_v_c(x_v, x_c), 1)
        logits = self.fc(self.fusion_r_q(x_r, x_q))
        
        return logits

    def get_question(self, data):
        n = data['image'].size(0)
        question_text = data['question']
        with torch.no_grad():
            question_input_ids = self.tokenizer.batch_encode_plus([x for x in question_text], return_tensors='pt')['input_ids'].to(self.device)
            question_encoded, _ = self.bert(question_input_ids)
            question_encoded = question_encoded.mean(1)
        return torch.tanh(self.linear_q(self.dropout_q(question_encoded)))

    def get_contexts(self, data):
        n = data['image'].size(0)
        contexts_text = data['contexts']
        with torch.no_grad():
            contexts_input_ids = self.tokenizer.batch_encode_plus([y for x in contexts_text for y in x], return_tensors='pt')['input_ids'].to(self.device)
            contexts_encoded, _ = self.bert(contexts_input_ids)
            contexts_encoded = contexts_encoded.mean(1)
        max_contexts_length = 5
        contexts = torch.zeros(max_contexts_length, n, self.dim_bert).to(self.device)
        idx = 0
        for i in range(n):
            contexts_length = len(contexts_text[i])
            contexts[:contexts_length, i, :] = contexts_encoded[idx:idx+contexts_length]
            idx += contexts_length
        _, (_, contexts) = self.lstm_c(contexts)
        contexts = self.dropout_c(contexts.mean(0))
        return torch.tanh(contexts)
