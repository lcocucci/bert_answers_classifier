import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# Para regresión num_labels=1
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "bert_regression_model.pt")
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

def continuous_to_discrete(value):
    # Ajusta los umbrales según la distribución de tu modelo
    if value < 1:
        return 0
    elif value < 3:
        return 2
    elif value < 5:
        return 4
    elif value < 7:
        return 6
    elif value < 9:
        return 8
    else:
        return 10

def predict_score(question, correct_answer, student_answer):
    input_text = f"[CLS] {question} [SEP] {correct_answer} [SEP] {student_answer} [SEP]"
    encodings = tokenizer(
        input_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"]
        )
    
    predicted_value = outputs.logits.item()  # valor continuo
    predicted_score = continuous_to_discrete(predicted_value)
    return predicted_score
