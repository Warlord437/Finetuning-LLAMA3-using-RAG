# Sentiment Analysis with Llama 3 and RAG

This repository contains a hands-on tutorial on fine-tuning a Llama 3 model for sentiment analysis on financial and economic information. The project demonstrates the use of advanced NLP techniques to gain insights into market trends, investor confidence, consumer behavior, and more.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Libraries and Tools](#libraries-and-tools)
- [Setup and Installation](#setup-and-installation)
- [Code Explanation](#code-explanation)
  - [Importing Libraries and Setting Environment Variables](#importing-libraries-and-setting-environment-variables)
  - [Data Preparation](#data-preparation)
  - [Evaluation Function](#evaluation-function)
  - [Predict Function](#predict-function)
  - [Fine-Tuning the Model](#fine-tuning-the-model)
  - [Setting Up RAG](#setting-up-rag)
  - [Example Usage](#example-usage)
- [Conclusion](#conclusion)
- [License](#license)

## Overview

Sentiment analysis on financial and economic information is crucial for businesses to make informed decisions. This tutorial fine-tunes a Llama 3 model using the FinancialPhraseBank dataset, which consists of financial news headlines annotated for sentiment (positive, neutral, or negative). The project also integrates Retrieval-Augmented Generation (RAG) to enhance the model's performance.

## Dataset

The FinancialPhraseBank dataset, introduced by scholars from the Aalto University School of Business in 2014, contains approximately 5000 sentences. Annotators assessed the sentences solely from an investor's perspective, evaluating the potential impact on stock prices.

## Libraries and Tools

The following libraries and tools are used in this project:

- [PyTorch](https://pytorch.org/): An open-source machine learning library for Python, primarily developed by Facebook's AI Research lab.
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes): A lightweight wrapper around CUDA custom functions, particularly for 8-bit optimizers and quantization functions.
- [Transformers](https://huggingface.co/transformers/): A library providing general-purpose architectures for natural language understanding (NLU) and natural language generation (NLG) with a large number of pre-trained models.
- [Accelerate](https://huggingface.co/docs/accelerate/index): A library for distributed training, enabling models to be trained on multiple GPUs or CPUs.
- [PEFT](https://github.com/huggingface/peft): A library for efficient adaptation of pre-trained language models to various downstream applications without fine-tuning all the model's parameters.
- [TRL](https://github.com/huggingface/trl): A library providing tools to train transformer language models with Reinforcement Learning.
- [Datasets](https://huggingface.co/docs/datasets/): A library for easily accessing and sharing datasets, with a focus on NLP.
- [SentenceTransformers](https://www.sbert.net/): A library for computing dense vector representations of sentences.
- [FAISS](https://github.com/facebookresearch/faiss): A library for efficient similarity search and clustering of dense vectors.

## Setup and Installation

To set up the environment and install the necessary libraries, run the following commands:

```bash
!pip install -q -U torch --index-url https://download.pytorch.org/whl/cu117
!pip install -q -U -i https://pypi.org/simple/ bitsandbytes
!pip install -q -U transformers=="4.40.0"
!pip install -q -U accelerate
!pip install -q -U datasets
!pip install -q -U trl
!pip install -q -U peft
!pip install -q -U tensorboard
!pip install -q -U faiss-cpu
!pip install -q -U sentence-transformers


Dataset Linked to Kaggle: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news

Kaggle Notebook: https://www.kaggle.com/code/jsaha437/fine-tune-llama-3-for-sentiment-analysis-5e8f87

