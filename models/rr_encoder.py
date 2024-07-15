import torch
from torch import nn
from transformers import BertModel, BertTokenizer

class RadiologyNotesEncoder(nn.Module):
    def __init__(self, pretrained_model_name='emilyalsentzer/Bio_ClinicalBERT', output_dim=384):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.output_dim = output_dim

    def forward(self, notes):
        # Tokenize input notes
        encoded_input = self.tokenizer(notes, padding=True, truncation=True, return_tensors='pt')
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']

        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token output

        # Project to desired output dimension
        v_rr = self.fc(cls_output)
        return v_rr