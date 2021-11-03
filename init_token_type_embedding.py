import torch
from torch import nn
from transformers import BertModel, BertConfig
model = BertModel.from_pretrained("/home/nlp/shaked571/ParaShoot/alephbert-base")

model.embeddings.token_type_embeddings =nn.Embedding.from_pretrained(torch.cat([model.embeddings.token_type_embeddings.weight, model.embeddings.token_type_embeddings.weight]))

torch.save(model.state_dict(), "/home/nlp/shaked571/ParaShoot/alephbert-base/model2.bin")
