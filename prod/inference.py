import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Cargar el tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Crear el modelo para regresión con num_labels=1
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)

# Cargar el state_dict del modelo entrenado
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "bert_regression_model.pt")

state_dict = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(state_dict, strict=True)
model.eval()

def predict_score(question, correct_answer, student_answer):
    # Tokenizar las entradas
    input_text = f"[CLS] {question} [SEP] {correct_answer} [SEP] {student_answer} [SEP]"
    encodings = tokenizer(
        input_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Inferencia sin gradientes
    with torch.no_grad():
        outputs = model(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"]
        )

    # `outputs.logits` ahora es un tensor [1,1] con un valor continuo (regresión)
    predicted_value = outputs.logits.item()

    # Si quieres devolver directamente el valor continuo:
    return predicted_value

    # Opcional: Si necesitas discretizar el valor continuo a un conjunto de puntajes
    # predefinidos (ej. 0,2,4,6,8,10), puedes definir una función:
    #
    # def continuous_to_discrete(value):
    #     if value < 1:
    #         return 0
    #     elif value < 3:
    #         return 2
    #     elif value < 5:
    #         return 4
    #     elif value < 7:
    #         return 6
    #     elif value < 9:
    #         return 8
    #     else:
    #         return 10
    #
    # return continuous_to_discrete(predicted_value)
