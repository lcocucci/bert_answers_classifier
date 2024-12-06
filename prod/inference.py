import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Cargagmos tokenizer y modelo (ajustamos el path según corresponda)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
num_classes = 6  # 6 categorías: correspondencia con puntajes {0,2,4,6,8,10}
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)

# Cargamos pesos entrenados locales
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "bert_regression_model.pt")
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

score_map = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8, 5: 10}

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

    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    predicted_score = score_map[predicted_class]
    return predicted_score
