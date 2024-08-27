Skim Literature by Pretrained Sentence Encoder.
 
The goal of this project is to create an NLP model that can convert an abstract of a research paper into different parts such as Background, Objective, Method, Result, or Conclusion as per the sentences to enable readers quickly and efficiently skim through the long literature.

The dataset is taken from this paper: https://arxiv.org/abs/1710.06071 which consists of approximately 20,000 abstracts of randomized controlled trials. Each sentence of each abstract is labeled with their role in the abstract using one of the following classes: background, objective, method, result, or conclusion. 

First of all, since the dataset was in the text file, I preprocessed the data to read each of the lines in the target file. Then, converted the preprocessed data into dataframe. After that, all the text input are One-Hot encoded to convert into numbers while the 5 target labels are Label Encoded. Next, I did a series of experimentations starting with a Naive-Bayes as a baseline. After that, I prepared the data for deep learning models by creating a Text Vectorization layer and an Embedding layer. To load the data faster, I turned the dataset into a Prefetch dataset of batches. Then, I built 5 different models.

> - Conv1D with token embeddings
> - feature extraction with pretrained Universal Sentence Encoder from TensorFlow Hub
> - Conv1D with character embeddings
> - Combined pretrained token embeddings + character embeddings (hybrid embedding layer)
> - Transfer Learning with pretrained token embeddings + character embeddings + positional embeddings (tribrid embedding layer)

Out of which, tribrid embedding model performed the best with 83.30 accuracy, 0.83 precision, 0.83 recall, and 0.83 f1-score on training set and 82.57 accuracy, 0.82 precision, 0.82 recall, and 0.82 f1-score on test set. Then, I found the most wrong predictions and made predictions on open example abstracts.
