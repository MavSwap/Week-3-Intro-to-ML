# Week-3-Intro-to-ML
**Overview**

This project implements a complete machine learning pipeline for sentiment analysis using Hugging Face's transformers and datasets libraries. The IMDb dataset, which contains labeled movie reviews, is used to fine-tune a pre-trained BERT model (bert-base-uncased).

**Pipeline Components**

**Data Loading**

The IMDb dataset is loaded using load_dataset from the Hugging Face Datasets library.

The dataset is already split into training and test sets, with binary labels for positive and negative reviews.

**Preprocessing**

Tokenization is performed using BertTokenizer from Hugging Face.

The tokenizer handles truncation and padding to prepare the text inputs for the BERT model.

**Model Setup**

A pre-trained bert-base-uncased model is used with a classification head for binary sentiment prediction.

The Trainer API is used to streamline the training process with defined hyperparameters.

**Training & Evaluation**

The model is trained on a subset of the IMDb training set (for efficiency).

Performance is evaluated using Accuracy and F1-score metrics to ensure robustness.

**Saving and Inference**

The fine-tuned model and tokenizer are saved locally.

We demonstrate how to reload the model and run sentiment inference on new text using the pipeline API.

**Design Rationale**

We selected the bert-base-uncased model because of its proven effectiveness in a wide range of NLP tasks. Using a pre-trained model significantly reduces training time while maintaining high accuracy. The Hugging Face Trainer simplifies the training, evaluation, and logging process, making it easier to manage experiments.

**Anticipated Challenges**

Computational Constraints: BERT-based models are resource-intensive. To accommodate limited hardware, we reduce the dataset size and training epochs.

Data Preprocessing: Ensuring correct tokenization and handling edge cases like long reviews are essential for model performance.

Overfitting: We use early stopping via load_best_model_at_end to mitigate overfitting during fine-tuning.
