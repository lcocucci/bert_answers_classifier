# inference.py
import torch
import torch.nn as nn
import os
from transformers import AutoTokenizer, AutoModel

# Mismo tokenizador y modelo base que en Colab
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

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
        output = self.regressor(pooled_output)
        return output.squeeze(-1)

# Cargar el modelo base igual que en Colab
bert_model = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

model = BertRegressionModel(bert_model)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "bert_regression_model.pt")

state_dict = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval()

def predict_score(question, correct_answer, student_answer):
    # Tokenizar como en el entrenamiento (usando text y text_pair)
    # En el entrenamiento se usaba:
    # inputs = tokenizer(
    #     text=respuestas_correctas,
    #     text_pair=respuestas_estudiantes,
    #     padding='longest',
    #     truncation=True,
    #     return_tensors='pt'
    # )
    #
    # Ahora haremos lo mismo para una sola muestra:
    inputs = tokenizer(
        text=[correct_answer],
        text_pair=[student_answer],
        padding='longest',
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        # El modelo espera input_ids, attention_mask, token_type_ids
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

    # 'outputs' es un valor continuo normalizado entre 0 y 1
    predicted_value = outputs.item()
    # Desnormalizar multiplicando por 10
    predicted_score = predicted_value * 10.0

    # Si quieres redondear o asignar a categorías discretas lo haces aquí.
    # Si no, devuélvelo continuo.
    return predicted_score
