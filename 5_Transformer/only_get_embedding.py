from transformers import AutoTokenizer, AutoModel
import torch

model_name = 'roberta-large'
num_layers = 17

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

assert (0 <= num_layers <= len(model.encoder.layer)), f"Invalid num_layers: num_layers should be between 0 and {len(model.encoder.layer)} for {model_name}"
model.encoder.layer = torch.nn.ModuleList([layer for layer in model.encoder.layer[:num_layers]])

onesent = 'Some sentence is here.'
onesent_tokens = tokenizer.encode(onesent)

model.eval()
with torch.no_grad():
  out = model(onesent_tokens, attention_mask=attention_mask)
emb = out[0]
emb
