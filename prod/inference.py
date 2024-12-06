import torch
import torch.nn as nn
import os
from transformers import AutoTokenizer, AutoModel

# Usa el mismo modelo base con el que se entrenó el checkpoint. 
# Si el checkpoint proviene de bert-base-uncased con 6 salidas, usa bert-base-uncased.
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class BertRegressionModel(nn.Module):
    def __init__(self, bert_model):
        super(BertRegressionModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(p=0.3)
        # Primero creamos la capa final con 6 salidas para cargar el checkpoint
        self.regressor = nn.Linear(self.bert.config.hidden_size, 6)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.regressor(pooled_output)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "bert_regression_model.pt")

bert_model = AutoModel.from_pretrained("bert-base-uncased")
model = BertRegressionModel(bert_model)

# Cargar el checkpoint original
state_dict = torch.load(model_path, map_location=torch.device("cpu"))

# Si en el checkpoint las capas finales se llamaban "classifier.*", las renombramos a "regressor.*"
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("classifier", "regressor")
    new_state_dict[new_key] = v

# Cargar el state_dict adaptado
model.load_state_dict(new_state_dict, strict=True)

# Ahora que cargamos con éxito (6 salidas), reemplazamos la capa final por una de 1 salida
old_weight = model.regressor.weight.data.clone()
old_bias = model.regressor.bias.data.clone()

# Redefinimos la capa final con 1 salida
model.regressor = nn.Linear(model.bert.config.hidden_size, 1)

# Inicializar la nueva capa usando una de las neuronas antiguas (por ejemplo la primera)
model.regressor.weight.data = old_weight[0:1, :]  # Copiamos solo la primera "neurona"
model.regressor.bias.data = old_bias[0:1]          # Copiamos el sesgo correspondiente

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

        # Ahora model(...) da salida [batch, 1]
        # Si el modelo originalmente generaba valores continuos normalizados 0-1
        # Aquí asumes que la primera neurona se corresponda con algo útil.
        # Si no, es un punto de partida (tendrás que reajustar según el caso real).
        normalized_score = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

    # Desnormalizar si correspondía. Supongamos que el original estaba normalizado
    predicted_score = normalized_score.item() * 10.0
    return predicted_score
