import torch

class Dataset(torch.utils.data.Dataset):
    """
    This class loads and preprocesses the given text data
    """
    def __init__(self, paths, tokenizer):
        """
        This function initialises the object. It takes the given paths and tokeniser.
        """
        # the last file might not have 10000 samples, which makes it difficult to get the total length of the ds
        self.paths = paths[:len(paths)-1]
        self.tokenizer = tokenizer
        self.data = self.read_file(self.paths[0])
        self.current_file = 1
        self.remaining = len(self.data)
        self.encodings = self.get_encodings(self.data)

    def __len__(self):
        """
        returns the lenght of the ds
        """
        return 10000*len(self.paths)
    
    def read_file(self, path):
        """
        reads a given file
        """
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        return lines

    def get_encodings(self, lines_all):
        """
        Creates encodings for a given text input
        """
        # tokenise all text 
        batch = self.tokenizer(lines_all, max_length=512, padding='max_length', truncation=True)

        # Ground Truth
        labels = torch.tensor(batch['input_ids'])
        # Attention Masks
        mask = torch.tensor(batch['attention_mask'])

        # Input to be masked
        input_ids = labels.detach().clone()
        rand = torch.rand(input_ids.shape)

        # with a probability of 15%, mask a given word, leave out CLS, SEP and PAD
        mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 2) * (input_ids != 3)
        # assign token 4 (=MASK)
        input_ids[mask_arr] = 4
        
        return {'input_ids':input_ids, 'attention_mask':mask, 'labels':labels}

    def __getitem__(self, i):
        """
        returns item i
        Note: do not use shuffling for this dataset
        """
        # if we have looked at all items in the file - take next
        if self.remaining == 0:
            self.data = self.read_file(self.paths[self.current_file])
            self.current_file += 1
            self.remaining = len(self.data)
            self.encodings = self.get_encodings(self.data)
        
        # if we are at the end of the dataset, start over again
        if self.current_file == len(self.paths):
            self.current_file = 0
                 
        self.remaining -= 1    
        return {key: tensor[i%10000] for key, tensor in self.encodings.items()}  

def test_model(model, optim, test_ds_loader, device):
    """
    This function tests whether the parameters of the model that are frozen change, the ones that are not frozen do change,
    and whether any parameters become NaN or Inf
    :param model: model to be tested
    :param optim: optimiser used for training
    :param test_ds_loader: dataset to perform the forward pass on
    :param device: current device
    :raises Exception: if any of the above conditions are not met
    """
    ## Check if non-frozen parameters changed and frozen ones did not

    # get initial parameters to check against
    params = [ np for np in model.named_parameters() if np[1].requires_grad ]
    initial_params = [ (name, p.clone()) for (name, p) in params ]

    params_frozen = [ np for np in model.named_parameters() if not np[1].requires_grad ]
    initial_params_frozen = [ (name, p.clone()) for (name, p) in params_frozen ]

    optim.zero_grad()

    # get data
    batch = next(iter(test_ds_loader))

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    # forward pass and backpropagation
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
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