import streamlit as st
from inference import predict_score

st.title("Evaluación de Respuestas Abiertas")
st.write("Ingrese la pregunta, la respuesta correcta y la respuesta del estudiante para obtener un puntaje.")

question = st.text_area("Pregunta", "")
correct_answer = st.text_area("Respuesta Correcta", "")
student_answer = st.text_area("Respuesta del Estudiante", "")

if st.button("Evaluar Respuesta"):
    if question and correct_answer and student_answer:
        score = predict_score(question, correct_answer, student_answer)
        st.write(f"**Puntaje asignado (valor continuo):** {score:.2f}")
    else:
        st.warning("Por favor, complete todos los campos antes de evaluar.")
