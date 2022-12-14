# Applied Deep Learning
- Sabrina Herbst
- Project Type: Bring your own method

## Structure
- `data/` contains the data used for the project (after running `load_data.py`, and downloading the natural questions)
- `distilbert.py` contains the code for the DistilBERT model and the Dataset. A function for testing the functionality is in there too.
- `distilbert.ipynb` contains the creation and training of the DistilBERT model
- `distilbert.model` is the distilbert model
- `distilbert_reuse.model` is the question answering model
- `load_data.py` contains the code for loading the data and preprocessing it. We also split it up into smaller files to load in the Dataset later on.
- `qa_model.py` contains the code for thee different QA models. We also define a separate Dataset class in there and a method for testing the models.
- `qa_model.ipynb` contains the creation and training of the QA models.
- `requirements.txt` contains the requirements for the project
- `utils.py` contains some helper functions for the project. It contains the functions to evaluate the models and a way to visualise the trained parameters for each model.
- `application.py` contains the streamlit application to run everything

## How to run
- Install the requirements with `pip install -r requirements.txt`
- Run `load_data.py` to download the data and preprocess it (follow the documentation in the file regarding the natural questions dataset)
- Run `distilbert.ipynb` to train the DistilBERT model
- Run `qa_model.ipynb` to train the QA models
- Run `streamlit run application.py` to run the streamlit app

## Project
1. Create own DistilBERT Model using the OpenWebText dataset from Huggingface (https://huggingface.co/datasets/openwebtext) - 20h (active work, training is a lot longer)
   - I initially wanted to use the Oscar dataset (https://huggingface.co/datasets/oscar), but it took too much storage
   - I will train a MaskedLM model myself. However, my computational resources are limiting me, so should my model's performance not be sufficient, I will fall back onto the Huggingface model (https://huggingface.co/distilbert-base-cased)
2. Current methods often fine-tune the models on specific tasks. I believe that MultiTask learning is extremely useful, hence, I want to fix the DistilBERT weights here and train a head to do question answering - 30h
   - Dataset: SQuAD (https://paperswithcode.com/dataset/squad), maybe also Natural Questions (https://paperswithcode.com/dataset/natural-questions)
   - The idea is to have one common corpus and specific heads, rather than a separate model for every single task
   - In particular, I want to evaluate whether it is really necessary to fine-tune the base model too, as it already contains a model of the language. Ideally, having task-specific heads could make up for the lacking fine-tuning of the base model.
   - If the performance of the model is comparable, this could reduce training efforts and resources 
   - Either add another Bert Layer per task or just the multi-head self-attention layer (see next section)
3. Application - 10h
   - GUI, that lets people enter a context (base text), question, and they will receive an answer.
   - Will contain some SQuAD questions as examples.
4. Report - 2h
5. Presentation - 2h
  
## Goal
The DistilBERT model was quite straightforward to train, I mostly used what HuggingFace provided anyways, so the only real challenge here was to download the dataset. Also, training is a lot of effort, so I wasn't able to train it to full convergence, as I just didn't have the resource. The DistilBERT model can be found in `distilbert.ipynb` and is fully functional.
* Error Metric: I landed at about 0.2 CrossEntropyLoss for both training and test set. The preconfiguration is quite good, as it didn't overfit. 
* DistilBERT is primarily trained for masked prediction, I ran some manual sanity tests, to see which words are predicted. They usually make sense (although not entirely sometimes) and the grammatics are usually quite correct too. 
  * e.g. "It seems important to tackle the climate [MSK]." gave change (19%), crisis (12%), issues (5.8%), which are all appropriate in the context.

Now for the Question Answering model.

* Error Metric:
  * We use the CrossEntropy loss to train the QA model
  * Afterwards, we will fall back to F-1 score and the Exact Match (EM). These are also the metrics used for the SQuAD competition. (https://rajpurkar.github.io/SQuAD-explorer/).
  * The definitions are retrieved from here (https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#Metrics-for-QA).
  * EM: 1 if the prediction exactly matches the original, 0 otherwise
  * F-1: Computed over the individual words in the prediction against those in the answer. Number of shared words is the key. Precision: Ratio of shared words to the number of words in the prediction. Recall: Ratio of shared words to number of words in GT.
* Target for Error Metric:
  * EM: 0.6
  * F-1: 0.7
* Achieved value: I almost achieved the target for both of the measurements. I ultimately quit I had already spent a lot of time on the project and thought that the results were reasonable.
  * EM: 0.52
  * F-1: 0.67

Amount of time for each task:
  * DistilBERT model: ~20h (without training time). This was very similar to what I estimated, because I relied heavily on the Huggingface library. Loading the data was easy and the data is already very clean.
  * QA model: ~40h (without training time). Was a lot of effort, as my first approach didn't work and it took me making up a basic POC model, to get to the final architecture.
  * Application: 2h. Streamlit was really easy to use and fairly straightforward.
  
## Data
- Aaron Gokaslan et al. OpenWebText Corpus. 2019. https://skylion007.github.io/OpenWebTextCorpus/: **OpenWebText**
  - Open source replication of the WebText dataset from OpenAI. 
  - They scraped web pages, with a focus on quality. They looked at the Reddit up- and downvotes to determine the quality of the resource. 
  - The dataset will be used to train the DistilBERT model using language masking.
- Rajpurkar et al. SQuAD: 100,000+ Questions for Machine Comprehension of Text. 2016. https://rajpurkar.github.io/SQuAD-explorer/): **SQuAD**
  - Standford Question Answering Dataset 
  - Collection of question-answer pairs, where the answer is a sequence of tokens in the given context text. 
  - Very diverse because it was created using crowdsourcing.
- Kwiatkowski et al. Natural Questions: a Benchmark for Question Answering Research. 2019. https://ai.google.com/research/NaturalQuestions/: **Natural Questions**
  - Also a question-answer set, based on a Google query and corresponding Wikipedia page, containing the answer. 
  - Very similar to the SQuAD dataset. 
- Yang, Zhilin et al. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. https://hotpotqa.github.io/
 
 
## Related Papers
- Sanh, Victor et al. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. ArXiv abs/1910.01108. 2019.: https://arxiv.org/abs/1910.01108v4
  - The choice of DistilBERT, as opposed to BERT, RoBERTa or XLNet is primarily based on the size of the network and training time
  - I hope that the slight performance degradation will be compensated by the head, that is fine-tuned
- Ł. Maziarka and T. Danel. Multitask Learning Using BERT with Task-Embedded Attention. 2021 International Joint Conference on Neural Networks (IJCNN). 2021, pp. 1-6: https://ieeexplore.ieee.org/document/9533990
  - In the paper they add task-specific parameters to the original model, hence, they change the baseline BERT
  - "One possible solution is to add the task-specific, randomly initialized BERT_LAYERS at the top of the model."
	- This is an interesting approach
	- However, it increases the parameters drastically
  - "We could prune the number of parameters in this setting, by adding only the multi-head self-attention layer, 
	without the position-wise feed-forward network."
	- This would also be an interesting approach to investigate
- Jia, Qinjin et al. ALL-IN-ONE: Multi-Task Learning BERT models for Evaluating Peer Assessments. ArXiV abs/2110.03895. 2021.: https://arxiv.org/abs/2110.03895
  - The authors compared single-task fine-tuned models (BERT and DistiLBERT) with multitask models 
  - They added one Dense layer on top of the base model for single-task, and three Dense layers for multitask
  - They did not fix the base model's weights though, instead they fine-tuned it on multiple tasks, adding up the cross-entropy for each task to create the loss function
- El Mekki et al. BERT-based Multi-Task Model for Country and Province Level MSA and Dialectal Arabic Identification. WANLP. 2021.: https://aclanthology.org/2021.wanlp-1.31/
  - The authors use a BERT (MARBERT), task specific attention layers and then classifiers to train the network
  - They do not fix the weights of the BERT model either
- Jia et al. Large-scale Transfer Learning for Low-resource Spoken Language Understanding. ArXiV abs/2008.05671. 2020.: https://arxiv.org/abs/2008.05671
  - This paper deals with Spoken Language Understanding (SLU)
  - The authors test an architecture, where they fine-tune the BERT model and one where they fix the weights and add a specific head on top
  - They conclude: "Results in Table 4 indicate that both strategies have abilities of improving the performance of SLU model."

