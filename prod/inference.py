import torch
import torch.nn as nn
import os
from transformers import AutoTokenizer, AutoModel

# Usa el mismo modelo base que en el entrenamiento (bert-base-uncased si el checkpoint tiene 30522 tokens)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class BertRegressionModel(nn.Module):
    def __init__(self, bert_model):
        super(BertRegressionModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(p=0.3)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.regressor(pooled_output).squeeze(-1)

# Carga el modelo base idéntico al usado en entrenamiento
bert_model = AutoModel.from_pretrained("bert-base-uncased")
model = BertRegressionModel(bert_model)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "bert_regression_model.pt")

# Carga el state dict original
state_dict = torch.load(model_path, map_location=torch.device("cpu"))

# Renombrar claves classifier.* a regressor.*
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("classifier."):
        # Remplaza 'classifier' por 'regressor'
        new_key = k.replace("classifier", "regressor")
        new_state_dict[new_key] = v
    else:
        # Otras claves se mantienen igual
        new_state_dict[k] = v

# Carga el nuevo state dict en el modelo
model.load_state_dict(new_state_dict, strict=True)
model.eval()

def predict_score(question, correct_answer, student_answer):
    inputs = tokenizer(
        text=[correct_answer],
        text_pair=[student_answer],
        padding='longest',
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        # El modelo produce un valor entre 0 y 1 si así fue entrenado
        normalized_score = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

    # Desnormaliza si fue entrenado así (0-1 a 0-10):
    predicted_score = normalized_score.item() * 10.0
    return predicted_score
