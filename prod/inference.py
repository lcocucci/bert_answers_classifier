import torch
import torch.nn as nn
import os
from transformers import AutoTokenizer, AutoModel

# Mismo modelo base que el utilizado en Colab
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
        # outputs.pooler_output es la representación [CLS]
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        output = self.regressor(pooled_output)  # [batch_size, 1]
        return output.squeeze(-1)  # [batch_size]

# Cargar el modelo base BETO igual que en el Colab
bert_model = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

# Instanciar el modelo de regresión
model = BertRegressionModel(bert_model)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "bert_regression_model.pt")

# Cargar el state_dict del modelo entrenado
state_dict = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(state_dict, strict=True)
model.eval()

def predict_score(question, correct_answer, student_answer):
    # Tokenizar usando text y text_pair, exactamente como en el entrenamiento.
    # No agregar manualmente [CLS] ni [SEP], el tokenizador se encarga.
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

        # El modelo produce un valor entre 0 y 1 (normalizado)
        normalized_score = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

    # Desnormalizar el puntaje a escala 0-10
    predicted_score = normalized_score.item() * 10.0
    return predicted_score
