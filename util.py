import re
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm
import torch


def normalize_text(s):
    """
    Removes articles and punctuation, and standardizing whitespace are all typical text processing steps.
    Copied from: https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#Metrics-for-QA
    :param s: string to clean
    :return: cleaned string
    """
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, truth):
    """
    Returns true if the predicted is an exact match, else False
    Retrieved from: https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#Metrics-for-QA
    :param prediction: predicted answer
    :param truth: ground truth
    :return: 1 if exact match, else 0
    """
    return int(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
    """
    Computes the F-1 score of a prediction, based on the tokens
    Retrieved from: https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#Metrics-for-QA
    :param prediction: predicted answer
    :param truth: ground truth
    :return: the f-1 score of the prediction
    """
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    # get tokens that are in the prediction and gt
    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    # calculate precision and recall
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)

def eval_test_set(model, tokenizer, test_loader, device):
    """
    Calculates the mean EM and mean F-1 score on the test set
    :param model: pytorch model
    :param tokenizer: tokenizer used to encode the samples
    :param test_loader: dataloader object with test data
    :param device: device the model is on
    """
    mean_em = []
    mean_f1 = []
    model.to(device)
    model.eval()
    for batch in tqdm(test_loader):
        # get test data and transfer to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start = batch['start_positions'].to(device)
        end = batch['end_positions'].to(device)

        # predict
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start, end_positions=end)

        # iterate over samples, calculate EM and F-1 for all
        for input_i, s, e, trues, truee in zip(input_ids, outputs['start_logits'], outputs['end_logits'], start, end):
            # get predicted start and end logits (maximum score)
            start_logits = torch.argmax(s)
            end_logits = torch.argmax(e)

            # get predicted answer as string
            ans_tokens = input_i[start_logits: end_logits + 1]
            answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
            predicted = tokenizer.convert_tokens_to_string(answer_tokens)

            # get ground truth as string
            ans_tokens = input_i[trues: truee + 1]
            answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
            true = tokenizer.convert_tokens_to_string(answer_tokens)

            # compute score
            em_score = compute_exact_match(predicted, true)
            f1_score = compute_f1(predicted, true)
            mean_em.append(em_score)
            mean_f1.append(f1_score)
    print("Mean EM: ", np.mean(mean_em))
    print("Mean F-1: ", np.mean(mean_f1))

def count_parameters(model):
    """
    This function prints statistic regarding the trainable parameters
    :param model: pytorch model
    :return: parameters to be fine-tuned
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
