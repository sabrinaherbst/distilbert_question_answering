from torch import nn
import torch
from typing import Optional
import copy
import pandas as pd

"""
This module contains the implementation of the QA model. We define three different models and a dataset class.
The structure is based on the Hugging Face implementations.
https://huggingface.co/docs/transformers/model_doc/distilbert
"""

class SimpleQuestionDistilBERT(nn.Module):
    """
    This class implements a simple version of the distilbert question answering model, following the implementation of Hugging Face, 
    https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/distilbert/modeling_distilbert.py#L805

    It basically fine-tunes a given distilbert model. We only add one linear layer on top, which determines the start and end logits.
    """
    def __init__(self, distilbert, dropout=0.1):
        """
        Creates and initialises model
        """
        super(SimpleQuestionDistilBERT, self).__init__()

        self.distilbert = distilbert

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(768, 2)

        # initialise weights
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.classifier.apply(init_weights)


    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                start_positions: Optional[torch.Tensor] = None,
                end_positions: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None):
        """
        This function implements the forward pass of the model. It takes the input_ids and attention_mask and returns the start and end logits.
        """
        # make predictions on base model
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # retrieve hidden states
        hidden_states = distilbert_output[0]  # (bs, max_query_len, dim)
        hidden_states = self.dropout(hidden_states)

        # make predictions on head
        logits = self.classifier(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1).contiguous()  # (bs, max_query_len)

        # calculate loss
        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return {"loss": total_loss,
                "start_logits": start_logits,
                "end_logits": end_logits,
                "hidden_states": distilbert_output.hidden_states,
                "attentions": distilbert_output.attentions}


class QuestionDistilBERT(nn.Module):
    """
    This class implements the distilbert question answering model. We fix all layers of the base model and only fine-tune the head.
    The head consists of a transformer encoder with three layers and a classifier on top.
    """
    def __init__(self, distilbert, dropout=0.1):
        """
        Creates and initialises QuestionDIstilBERT instance
        """
        super(QuestionDistilBERT, self).__init__()

        # fix parameters for base model
        for param in distilbert.parameters():
            param.requires_grad = False

        self.distilbert = distilbert
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)
        self.te = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=768, nhead=12), num_layers=3)

        # create custom head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        # initialise weights of the linear layers
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.classifier.apply(init_weights)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                start_positions: Optional[torch.Tensor] = None,
                end_positions: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None):
        """
        This function implements the forward pass of the model. It takes the input_ids and attention_mask and returns the start and end logits.
        """
        # make predictions on base model
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # retrieve hidden states
        hidden_states = distilbert_output[0]  # (bs, max_query_len, dim)
        hidden_states = self.dropout(hidden_states)
        attn_output = self.te(hidden_states)

        # make predictions on head
        logits = self.classifier(attn_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous() 
        end_logits = end_logits.squeeze(-1).contiguous()

        # calculate loss
        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return {"loss": total_loss,
                "start_logits": start_logits,
                "end_logits": end_logits,
                "hidden_states": distilbert_output.hidden_states,
                "attentions": distilbert_output.attentions}


class ReuseQuestionDistilBERT(nn.Module):
    """
    This class imports a model where all layers of the base distilbert model are fixed.
    Instead of training a completely new head, we copy the last two layers of the base model and add a classifier on top.
    """
    def __init__(self, distilbert, dropout=0.15):
        """
        Creates and initialises QuestionDIstilBERT instance
        """
        super(ReuseQuestionDistilBERT, self).__init__()
        self.te = copy.deepcopy(list(list(distilbert.children())[1].children())[0][-2:])
        # fix parameters for base model
        for param in distilbert.parameters():
            param.requires_grad = False

        self.distilbert = distilbert
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        # create custom head
        self.classifier = nn.Linear(768, 2)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.classifier.apply(init_weights)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                start_positions: Optional[torch.Tensor] = None,
                end_positions: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None):
        """
        This function implements the forward pass of the model. It takes the input_ids and attention_mask and returns the start and end logits.
        """
        # make predictions on base model
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # retrieve hidden states
        hidden_states = distilbert_output[0]  # (bs, max_query_len, dim)
        hidden_states = self.dropout(hidden_states)
        for te in self.te:
            hidden_states = te(
                x=hidden_states,
                attn_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions
            )[0]
        hidden_states = self.dropout(hidden_states)

        # make predictions on head
        logits = self.classifier(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1).contiguous()  # (bs, max_query_len)

        # calculate loss
        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return {"loss": total_loss,
                "start_logits": start_logits,
                "end_logits": end_logits,
                "hidden_states": distilbert_output.hidden_states,
                "attentions": distilbert_output.attentions}

class Dataset(torch.utils.data.Dataset):
    """
    This class creates a dataset for the DistilBERT qa-model.
    """
    def __init__(self, squad_paths, natural_question_paths, hotpotqa_paths, tokenizer):
        """
        creates and initialises dataset object
        """
        self.paths = []
        self.count = 0
        if squad_paths != None:
            self.paths.extend(squad_paths[:len(squad_paths)-1])
        if natural_question_paths != None:
            self.paths.extend(natural_question_paths[:len(natural_question_paths)-1])
        if hotpotqa_paths != None:
            self.paths.extend(hotpotqa_paths[:len(hotpotqa_paths)-1])
        self.data = None
        self.current_file = 0
        self.remaining = 0
        self.encodings = None
        # tokenizer for strings
        self.tokenizer = tokenizer
        
        
    def __len__(self):
        """
        returns the length of the dataset
        """
        return len(self.paths)*1000
    
    def read_file(self, path):
        """
        reads the file stored at path
        """
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        return lines

    def get_encodings(self):
        """
        returns encoded strings for the model
        """
        # remove leading and ending whitespaces
        questions = [q.strip() for q in self.data["question"]]
        context = [q.lower() for q in self.data["context"]]
    
        # tokenises questions and context. If the context is too long, we truncate it.
        inputs = self.tokenizer(
            questions,
            context,
            max_length=512,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        # tuples of integers giving us the original positions
        offset_mapping = inputs.pop("offset_mapping")
        
        answers = self.data["answer"]
        answer_start = self.data["answer_start"]

        # store beginning and end positions
        start_positions = []
        end_positions = []

        # iterate through questions
        for i, offset in enumerate(offset_mapping):

            answer = answers[i]
            start_char = int(answer_start[i])
            end_char = start_char + len(answer)

            sequence_ids = inputs.sequence_ids(i)

            # start and end of context based on tokens
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1

            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1
            
            # If answer not inside context add (0,0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
                self.count += 1
            else:
                # go to first offset position that is smaller than start char
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1

                start_positions.append(idx - 1)
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        # append start and end position to the embeddings
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        # return input_ids, attention mask, start and end positions (GT)
        return {'input_ids': torch.tensor(inputs['input_ids']), 
                'attention_mask': torch.tensor(inputs['attention_mask']),
                'start_positions': torch.tensor(inputs['start_positions']), 
                'end_positions': torch.tensor(inputs['end_positions'])}

    def __getitem__(self, i):
        """
        returns encoding of item i
        """
        
        # if we have looked at all items in the file - take next
        if self.remaining == 0:
            self.data = self.read_file(self.paths[self.current_file])
            self.data = pd.DataFrame([line.split("\t") for line in self.data], 
                                 columns=["context", "question", "answer", "answer_start"])
            self.current_file += 1
            self.remaining = len(self.data)
            self.encodings = self.get_encodings()
        # if we are at the end of the dataset, start over again
        if self.current_file == len(self.paths):
            self.current_file = 0
        self.remaining -= 1
        return {key: tensor[i%1000] for key, tensor in self.encodings.items()}

def test_model(model, optim, test_ds_loader, device):
    """
    This function is used to test the model's functionality, namely if params are not NaN and infinite,
    not-frozen parameters have to change, frozen ones must not
    :param model: pytorch model to evaluate
    :param optim: optimizer
    :param test_ds_loader: dataloader object
    :param device: device, the model is on
    :raises Exception if the model doesn't work as expected
    """
    ## Check if non-frozen parameters changed and frozen ones did not

    # get parameters used for tuning and store initial weight
    params = [np for np in model.named_parameters() if np[1].requires_grad]
    initial_params = [(name, p.clone()) for (name, p) in params]

    # get frozen parameters and store initial weight
    params_frozen = [np for np in model.named_parameters() if not np[1].requires_grad]
    initial_params_frozen = [(name, p.clone()) for (name, p) in params_frozen]

    # perform one iteration
    optim.zero_grad()
    batch = next(iter(test_ds_loader))

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    start_positions = batch['start_positions'].to(device)
    end_positions = batch['end_positions'].to(device)

    # forward pass and backpropagation
    outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                    end_positions=end_positions)
    loss = outputs['loss']
    loss.backward()
    optim.step()

    # check if variables have changed
    for (_, p0), (name, p1) in zip(initial_params, params):
        # check different than initial
        try:
            assert not torch.equal(p0.to(device), p1.to(device))
        except AssertionError:
            raise Exception(
                "{var_name} {msg}".format(
                    var_name=name,
                    msg='did not change!'
                )
            )
        # check not NaN
        try:
            assert not torch.isnan(p1).byte().any()
        except AssertionError:
            raise Exception(
                "{var_name} {msg}".format(
                    var_name=name,
                    msg='is NaN!'
                )
            )
        # check finite
        try:
            assert torch.isfinite(p1).byte().all()
        except AssertionError:
            raise Exception(
                "{var_name} {msg}".format(
                    var_name=name,
                    msg='is Inf!'
                )
            )

    # check that frozen weights have not changed
    for (_, p0), (name, p1) in zip(initial_params_frozen, params_frozen):
        # should be the same
        try:
            assert torch.equal(p0.to(device), p1.to(device))
        except AssertionError:
            raise Exception(
                "{var_name} {msg}".format(
                    var_name=name,
                    msg='changed!'
                )
            )
        # check not NaN
        try:
            assert not torch.isnan(p1).byte().any()
        except AssertionError:
            raise Exception(
                "{var_name} {msg}".format(
                    var_name=name,
                    msg='is NaN!'
                )
            )

        # check finite numbers
        try:
            assert torch.isfinite(p1).byte().all()
        except AssertionError:
            raise Exception(
                "{var_name} {msg}".format(
                    var_name=name,
                    msg='is Inf!'
                )
            )
    print("Passed")
