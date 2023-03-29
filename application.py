import streamlit as st
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForMaskedLM

from qa_model import ReuseQuestionDistilBERT

@st.cache(allow_output_mutation=True)
def load_model():
    mod = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased").distilbert
    m = ReuseQuestionDistilBERT(mod)
    m.load_state_dict(torch.load("distilbert_reuse.model", map_location=torch.device('cpu')))
    model = m
    del mod
    del m
    tokenizer = DistilBertTokenizer.from_pretrained('qa_tokenizer')
    return model, tokenizer


def get_answer(question, text, tokenizer, model):
    question = [question.strip()]
    text = [text.strip()]

    inputs = tokenizer(
        question,
        text,
        max_length=512,
        truncation="only_second",
        padding="max_length",
    )
    input_ids = torch.tensor(inputs['input_ids'])
    outputs = model(input_ids, attention_mask=torch.tensor(inputs['attention_mask']), start_positions=None, end_positions=None)

    start = torch.argmax(outputs['start_logits'])
    end = torch.argmax(outputs['end_logits'])

    ans_tokens = input_ids[0][start: end + 1]

    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
    predicted = tokenizer.convert_tokens_to_string(answer_tokens)
    return predicted


def main():
    st.set_page_config(page_title="Question Answering Tool", page_icon=":mag_right:")

    st.write("# Question Answering Tool \n"
         "This tool will help you find answers to your questions about the text you provide. \n"
         "Please enter your question and the text you want to search in the boxes below.")
    model, tokenizer = load_model()

    with st.form("qa_form"):
        # define a streamlit textarea
        text = st.text_area("Enter your text here", on_change=None)

        # define a streamlit input
        question = st.text_input("Enter your question here")
        
        if st.form_submit_button("Submit"):
            data_load_state = st.text('Let me think about that...')
            # call the function to get the answer
            answer = get_answer(question, text, tokenizer, model)
            # display the answer
            if answer == "":
                data_load_state.text("Sorry but I don't know the answer to that question")
            else:
                data_load_state.text(answer)


main()