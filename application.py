import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DistilBertForMaskedLM

from qa_model import ReuseQuestionDistilBERT

mod = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased").distilbert
m = ReuseQuestionDistilBERT(mod)
m.load_state_dict(torch.load("distilbert_reuse.model"))
model = m
del mod
del m
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

st.write("# Question Answering Tool \n"
         "This tool will help you find answers to your questions about the text you provide. \n"
         "Please enter your question and the text you want to search in the boxes below.")

# define a streamlit textarea
question = st.text_area("Enter your text here")

# define a streamlit input
text = st.text_input("Enter your question here")

# define a streamlit submit button
def get_answer(question, text):
    question = question.strip()
    text = text.strip()

    inputs = tokenizer(
        question,
        text,
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    outputs = model(torch.tensor(inputs['input_ids']), attention_mask=torch.tensor(inputs['attention_mask']), start_positions=None, end_positions=None)
    print(outputs)


if st.button("Submit"):
    # call the function to get the answer
    answer = get_answer(question, text)
    # display the answer
    st.write(answer)