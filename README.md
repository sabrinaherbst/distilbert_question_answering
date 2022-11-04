# Applied Deep Learning
- Sabrina Herbst
- Project Type: Bring your own method

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
  
## Goal
As the DistilBERT model was difficult to train, I will focus on the Question Answering model for the following. The DistilBERT model can be found in `distilbert.ipynb` and is fully functional, still, training required too many resources.

* Error Metric:
  * We use the CrossEntropy loss to train the QA model
  * Afterwards, we will fall back to F-1 score and the Exact Match (EM). These are also the metrics used for the SQuAD competition. (https://rajpurkar.github.io/SQuAD-explorer/).
  * The definitions are retrieved from here (https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#Metrics-for-QA).
  * EM: 1 if the prediction exactly matches the original, 0 otherwise
  * F-1: Computed over the individual words in the prediction against those in the answer. Number of shared words is the key. Precision: Ratio of shared words to the number of words in the prediction. Recall: Ratio of shared words to number of words in GT.
* Target for Error Metric:
  * Currently about rank 80 in the leaderboard.
  * EM: 0.7
  * F-1: 0.75
* Achieved value:
  * EM: WIP
  * F-1: WIP
* Amount of time for each task:
  * DistilBERT model: ~20h (without training time). This was very similar to what I estimated, because I relied heavily on the Huggingface library. Loading the data was easy and the data is already very clean.
  * QA model: WIP
  * Application: 
  * Report:
  * Presentation: