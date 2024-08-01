# %% [markdown]
# ## Fine-tune Llama 3 for Sentiment Analysis
# 
# For this hands-on tutorial on fine-tuning a Llama 3 model, I am going to deal with a sentiment analysis on financial and economic information. Sentiment analysis on financial and economic information is highly relevant for businesses for several key reasons, ranging from market insights (gain valuable insights into market trends, investor confidence, and consumer behavior) to risk management (identifying potential reputational risks) to investment decisions (gauging the sentiment of stakeholders, investors, and the general public businesses can assess the potential success of various investment opportunities).
# 
# Before the technicalities of fine-tuning a large language model like Llama 3, we have to find the correct dataset to demonstrate the potentialities of fine-tuning.
# 
# Particularly within the realm of finance and economic texts, annotated datasets are notably rare, with many being exclusively reserved for proprietary purposes. To address the issue of insufficient training data, scholars from the Aalto University School
# of Business introduced in 2014 a set of approximately 5000 sentences. This collection aimed to establish human-annotated benchmarks, serving as a standard for evaluating alternative modeling techniques. The involved annotators (16 people with
# adequate background knowledge on financial markets) were instructed to assess the sentences solely from the perspective of an investor, evaluating whether the news potentially holds a positive, negative, or neutral impact on the stock price.
# 
# The FinancialPhraseBank dataset is a comprehensive collection that captures the sentiments of financial news headlines from the viewpoint of a retail investor. Comprising two key columns, namely "Sentiment" and "News Headline," the dataset effectively classifies sentiments as either negative, neutral, or positive. This structured dataset serves as a valuable resource for analyzing and understanding the complex dynamics of sentiment in the domain of financial news. It has been used in various studies and research initiatives, since its inception in the work by Malo, P., Sinha, A., Korhonen, P., Wallenius, J., and Takala, P.  "Good debt or bad debt: Detecting semantic orientations in economic texts.", published in the Journal of the Association for Information Science and Technology in 2014.

# %% [markdown]
# As a first step, we install the specific libraries necessary to make this example work.

# %% [markdown]
# * accelerate is a distributed training library for PyTorch by HuggingFace. It allows you to train your models on multiple GPUs or CPUs in parallel (distributed configurations), which can significantly speed up training in presence of multiple GPUs (we won't use it in our example).
# * peft is a Python library by HuggingFace for efficient adaptation of pre-trained language models (PLMs) to various downstream applications without fine-tuning all the model's parameters. PEFT methods only fine-tune a small number of (extra) model parameters, thereby greatly decreasing the computational and storage costs.
# * bitsandbytes by Tim Dettmers, is a lightweight wrapper around CUDA custom functions, in particular 8-bit optimizers, matrix multiplication (LLM.int8()), and quantization functions. It allows to run models stored in 4-bit precision: while 4-bit bitsandbytes stores weights in 4-bits, the computation still happens in 16 or 32-bit and here any combination can be chosen (float16, bfloat16, float32, and so on).
# * transformers is a Python library for natural language processing (NLP). It provides a number of pre-trained models for NLP tasks such as text classification, question answering, and machine translation.
# * trl is a full stack library by HuggingFace providing a set of tools to train transformer language models with Reinforcement Learning, from the Supervised Fine-tuning step (SFT), Reward Modeling step (RM) to the Proximal Policy Optimization (PPO) step.

# %% [markdown]
# ## Installations and imports

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T02:33:11.492279Z","iopub.execute_input":"2024-08-01T02:33:11.493050Z","iopub.status.idle":"2024-08-01T02:36:17.779571Z","shell.execute_reply.started":"2024-08-01T02:33:11.493019Z","shell.execute_reply":"2024-08-01T02:36:17.778399Z"},"jupyter":{"outputs_hidden":false}}
!pip install -q -U torch --index-url https://download.pytorch.org/whl/cu117
!pip install -q -U -i https://pypi.org/simple/ bitsandbytes
!pip install -q -U transformers=="4.40.0"
!pip install -q -U accelerate
!pip install -q -U datasets
!pip install -q -U trl
!pip install -q -U peft
!pip install -q -U tensorboard
!pip install -q -U datasets
!pip install -q -U faiss-cpu
!pip install -q -U sentence-transformers
!pip install -q -U faiss-cpu sentence-transformers
!pip install -q -U accelerate

# %% [markdown]
# The code imports the os module and sets two environment variables:
# * CUDA_VISIBLE_DEVICES: This environment variable tells PyTorch which GPUs to use. In this case, the code is setting the environment variable to 0, which means that PyTorch will use the first GPU.
# * TOKENIZERS_PARALLELISM: This environment variable tells the Hugging Face Transformers library whether to parallelize the tokenization process. In this case, the code is setting the environment variable to false, which means that the tokenization process will not be parallelized.

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T02:36:17.782457Z","iopub.execute_input":"2024-08-01T02:36:17.782865Z","iopub.status.idle":"2024-08-01T02:36:35.473377Z","shell.execute_reply.started":"2024-08-01T02:36:17.782830Z","shell.execute_reply":"2024-08-01T02:36:35.472432Z"},"jupyter":{"outputs_hidden":false}}
import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    RagTokenizer, 
    RagRetriever, 
    RagSequenceForGeneration
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T02:36:35.474533Z","iopub.execute_input":"2024-08-01T02:36:35.475259Z","iopub.status.idle":"2024-08-01T02:36:35.479906Z","shell.execute_reply.started":"2024-08-01T02:36:35.475231Z","shell.execute_reply":"2024-08-01T02:36:35.478745Z"},"jupyter":{"outputs_hidden":false}}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %% [markdown]
# The code import warnings; warnings.filterwarnings("ignore") imports the warnings module and sets the warning filter to ignore. This means that all warnings will be suppressed and will not be displayed. Actually during training there are many warnings that do not prevent the fine-tuning but can be distracting and make you wonder if you are doing the correct things.

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T02:36:35.481169Z","iopub.execute_input":"2024-08-01T02:36:35.481497Z","iopub.status.idle":"2024-08-01T02:36:35.506684Z","shell.execute_reply.started":"2024-08-01T02:36:35.481464Z","shell.execute_reply":"2024-08-01T02:36:35.505856Z"},"jupyter":{"outputs_hidden":false}}

warnings.filterwarnings("ignore")

# %% [markdown]
# In the following cell there are all the other imports for running the notebook

# %% [code] {"papermill":{"duration":14.485002,"end_time":"2023-10-16T11:00:18.917449","exception":false,"start_time":"2023-10-16T11:00:04.432447","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-08-01T02:36:35.509496Z","iopub.execute_input":"2024-08-01T02:36:35.509931Z","iopub.status.idle":"2024-08-01T02:36:35.657711Z","shell.execute_reply.started":"2024-08-01T02:36:35.509905Z","shell.execute_reply":"2024-08-01T02:36:35.656995Z"},"jupyter":{"outputs_hidden":false}}

import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from trl import setup_chat_format
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline, 
                          logging, 
                          AutoModelForSeq2SeqLM, 
                          AutoTokenizer, 
                          RagTokenizer, 
                          RagRetriever, 
                          RagSequenceForGeneration)
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)
from sklearn.model_selection import train_test_split

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T02:36:35.658830Z","iopub.execute_input":"2024-08-01T02:36:35.659106Z","iopub.status.idle":"2024-08-01T02:36:35.664244Z","shell.execute_reply.started":"2024-08-01T02:36:35.659082Z","shell.execute_reply":"2024-08-01T02:36:35.663357Z"},"jupyter":{"outputs_hidden":false}}
print(f"pytorch version {torch.__version__}")

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T02:36:35.665378Z","iopub.execute_input":"2024-08-01T02:36:35.665665Z","iopub.status.idle":"2024-08-01T02:36:35.675899Z","shell.execute_reply.started":"2024-08-01T02:36:35.665642Z","shell.execute_reply":"2024-08-01T02:36:35.674947Z"},"jupyter":{"outputs_hidden":false}}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"working on {device}")

# %% [markdown]
# Disabling two features in PyTorch related to memory efficiency and speed during operations on the Graphics Processing Unit (GPU) specifically for the scaled dot product attention (SDPA) function.

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T02:36:35.677011Z","iopub.execute_input":"2024-08-01T02:36:35.677273Z","iopub.status.idle":"2024-08-01T02:36:35.684370Z","shell.execute_reply.started":"2024-08-01T02:36:35.677251Z","shell.execute_reply":"2024-08-01T02:36:35.683620Z"},"jupyter":{"outputs_hidden":false}}
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

# %% [markdown]
# ## Preparing the data and the core evaluation functions

# %% [markdown]
# The code in the next cell performs the following steps:
# 
# 1. Reads the input dataset from the all-data.csv file, which is a comma-separated value (CSV) file with two columns: sentiment and text.
# 2. Splits the dataset into training and test sets, with 300 samples in each set. The split is stratified by sentiment, so that each set contains a representative sample of positive, neutral, and negative sentiments.
# 3. Shuffles the train data in a replicable order (random_state=10)
# 4. Transforms the texts contained in the train and test data into prompts to be used by Llama: the train prompts contains the expected answer we want to fine-tune the model with
# 5. The residual examples not in train or test, for reporting purposes during training (but it won't be used for early stopping), is treated as evaluation data, which is sampled with repetition in order to have a 50/50/50 sample (negative instances are very few, hence they should be repeated)
# 5. The train and eval data are wrapped by the class from Hugging Face (https://huggingface.co/docs/datasets/index)
# 
# This prepares in a single cell train_data, eval_data and test_data datasets to be used in our fine tuning.

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T02:36:35.685419Z","iopub.execute_input":"2024-08-01T02:36:35.685715Z","iopub.status.idle":"2024-08-01T02:36:35.858938Z","shell.execute_reply.started":"2024-08-01T02:36:35.685681Z","shell.execute_reply":"2024-08-01T02:36:35.858066Z"},"jupyter":{"outputs_hidden":false}}
# Load the dataset
filename = "/kaggle/input/sentiment-analysis-for-financial-news/all-data.csv"
df = pd.read_csv(filename, names=["sentiment", "text"], encoding="utf-8", encoding_errors="replace", sep=",")
X_train, X_eval = [], []

for sentiment in ["positive", "neutral", "negative"]:
    train, test = train_test_split(df[df.sentiment == sentiment], train_size=0.7, test_size=0.3, random_state=42)
    X_train.append(train)
    X_eval.append(test)

X_train = pd.concat(X_train).sample(frac=1, random_state=10)
X_eval = pd.concat(X_eval).reset_index(drop=True)

def generate_prompt(data_point):
    return f"Text: '{data_point['text']}'\nSentiment: {data_point['sentiment']}"

def generate_test_prompt(data_point):
    return f"Text: '{data_point['text']}'\nSentiment: "

X_train = pd.DataFrame(X_train.apply(generate_prompt, axis=1), columns=["text"])
X_eval = pd.DataFrame(X_eval.apply(generate_test_prompt, axis=1), columns=["text"])
X_eval["sentiment"] = df["sentiment"]
train_data = Dataset.from_pandas(X_train)
eval_data = Dataset.from_pandas(X_eval)

print(X_train.shape, X_eval.shape)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T02:36:35.860168Z","iopub.execute_input":"2024-08-01T02:36:35.860549Z","iopub.status.idle":"2024-08-01T02:36:35.865987Z","shell.execute_reply.started":"2024-08-01T02:36:35.860516Z","shell.execute_reply":"2024-08-01T02:36:35.865161Z"},"jupyter":{"outputs_hidden":false}}
print(X_eval.shape)
print(X_train.shape)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T02:36:35.867134Z","iopub.execute_input":"2024-08-01T02:36:35.867459Z","iopub.status.idle":"2024-08-01T02:36:35.875048Z","shell.execute_reply.started":"2024-08-01T02:36:35.867427Z","shell.execute_reply":"2024-08-01T02:36:35.874206Z"},"jupyter":{"outputs_hidden":false}}
print(X_eval.iloc[0, 0])
# print(y_true.iloc[0])

print(X_train.shape)

# %% [markdown]
# Next we create a function to evaluate the results from our fine-tuned sentiment model. The function performs the following steps:
# 
# 1. Maps the sentiment labels to a numerical representation, where 2 represents positive, 1 represents neutral, and 0 represents negative.
# 2. Calculates the accuracy of the model on the test data.
# 3. Generates an accuracy report for each sentiment label.
# 4. Generates a classification report for the model.
# 5. Generates a confusion matrix for the model.

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T02:36:35.876031Z","iopub.execute_input":"2024-08-01T02:36:35.876292Z","iopub.status.idle":"2024-08-01T02:36:35.883648Z","shell.execute_reply.started":"2024-08-01T02:36:35.876270Z","shell.execute_reply":"2024-08-01T02:36:35.882781Z"},"jupyter":{"outputs_hidden":false}}
def predict(test, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["text"]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens=5, 
                        temperature=0.0)
        result = pipe(prompt)
        generated_text = result[0]['generated_text']
        answer = generated_text.split("Sentiment:")[-1].strip()
        if "positive" in answer:
            y_pred.append("positive")
        elif "negative" in answer:
            y_pred.append("negative")
        else:
            y_pred.append("neutral")
    return y_pred

# %% [markdown]
# ## Testing the model without fine-tuning

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T02:36:35.884921Z","iopub.execute_input":"2024-08-01T02:36:35.885320Z","iopub.status.idle":"2024-08-01T02:37:00.877711Z","shell.execute_reply.started":"2024-08-01T02:36:35.885290Z","shell.execute_reply":"2024-08-01T02:37:00.876474Z"},"jupyter":{"outputs_hidden":false}}
!pip install -q -U accelerate
!pip install -q -U -i https://pypi.org/simple/ bitsandbytes

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T02:37:00.883166Z","iopub.execute_input":"2024-08-01T02:37:00.883484Z","iopub.status.idle":"2024-08-01T02:39:14.774518Z","shell.execute_reply.started":"2024-08-01T02:37:00.883456Z","shell.execute_reply":"2024-08-01T02:39:14.773619Z"},"jupyter":{"outputs_hidden":false}}
model_name = "../input/llama-3/transformers/8b-chat-hf/1"

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    torch_dtype=compute_dtype,
    quantization_config=bnb_config, 
)

model.config.use_cache = False
model.config.pretraining_tp = 1

max_seq_length = 2048
tokenizer = AutoTokenizer.from_pretrained(model_name, max_seq_length=max_seq_length)
tokenizer.pad_token_id = tokenizer.eos_token_id

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T02:39:14.775604Z","iopub.execute_input":"2024-08-01T02:39:14.775891Z","iopub.status.idle":"2024-08-01T02:39:14.782804Z","shell.execute_reply.started":"2024-08-01T02:39:14.775865Z","shell.execute_reply":"2024-08-01T02:39:14.781902Z"},"jupyter":{"outputs_hidden":false}}
def predict(test, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["text"]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens=5, 
                        temperature=0.0)
        result = pipe(prompt)
        generated_text = result[0]['generated_text']
        answer = generated_text.split("Sentiment:")[-1].strip()
        if "positive" in answer:
            y_pred.append("positive")
        elif "negative" in answer:
            y_pred.append("negative")
        else:
            y_pred.append("neutral")
    return y_pred

# %% [markdown]
# At this point, we are ready to test the Llama 3 8b-chat-hf model and see how it performs on our problem without any fine-tuning. This allows us to get insights on the model itself and establish a baseline.

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T02:39:14.783957Z","iopub.execute_input":"2024-08-01T02:39:14.784289Z","iopub.status.idle":"2024-08-01T02:40:51.054270Z","shell.execute_reply.started":"2024-08-01T02:39:14.784260Z","shell.execute_reply":"2024-08-01T02:40:51.053362Z"},"jupyter":{"outputs_hidden":false}}
test.shape
y_pred = predict(test, model, tokenizer)

# %% [markdown]
# In the following cell, we evaluate the results. There is little to be said, it is performing really terribly because the 8b-chat-hf model tends to just predict a neutral sentiment and seldom it detects positive or negative sentiment.

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T02:40:51.055512Z","iopub.execute_input":"2024-08-01T02:40:51.055860Z","iopub.status.idle":"2024-08-01T02:40:51.064395Z","shell.execute_reply.started":"2024-08-01T02:40:51.055829Z","shell.execute_reply":"2024-08-01T02:40:51.063450Z"},"jupyter":{"outputs_hidden":false}}
def evaluate(y_true, y_pred):
    labels = ['positive', 'neutral', 'negative']
    mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
    
    y_true = np.vectorize(mapping.get)(y_true)
    y_pred = np.vectorize(mapping.get)(y_pred)
    
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')
    
    for label in set(y_true):
        label_indices = [i for i in range(len(y_true)) if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {accuracy:.3f}')
        
    print('\nClassification Report:')
    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=labels))
    
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2]))

# %% [markdown]
# **FINE TUNING**
# 
# In the next cell we set everything ready for the fine-tuning. We configures and initializes a Simple Fine-tuning Trainer (SFTTrainer) for training a large language model using the Parameter-Efficient Fine-Tuning (PEFT) method, which should save time as it operates on a reduced number of parameters compared to the model's overall size. The PEFT method focuses on refining a limited set of (additional) model parameters, while keeping the majority of the pre-trained LLM parameters fixed. This significantly reduces both computational and storage expenses. Additionally, this strategy addresses the challenge of catastrophic forgetting, which often occurs during the complete fine-tuning of LLMs.
# 
# PEFTConfig:
# 
# The peft_config object specifies the parameters for PEFT. The following are some of the most important parameters:
# 
# lora_alpha: The learning rate for the LoRA update matrices.
# lora_dropout: The dropout probability for the LoRA update matrices.
# r: The rank of the LoRA update matrices.
# bias: The type of bias to use. The possible values are none, additive, and learned.
# task_type: The type of task that the model is being trained for. The possible values are CAUSAL_LM and MASKED_LM.
# TrainingArguments:
# 
# The training_arguments object specifies the parameters for training the model. The following are some of the most important parameters:
# 
# output_dir: The directory where the training logs and checkpoints will be saved.
# num_train_epochs: The number of epochs to train the model for.
# per_device_train_batch_size: The number of samples in each batch on each device.
# gradient_accumulation_steps: The number of batches to accumulate gradients before updating the model parameters.
# optim: The optimizer to use for training the model.
# save_steps: The number of steps after which to save a checkpoint.
# logging_steps: The number of steps after which to log the training metrics.
# learning_rate: The learning rate for the optimizer.
# weight_decay: The weight decay parameter for the optimizer.
# fp16: Whether to use 16-bit floating-point precision.
# bf16: Whether to use BFloat16 precision.
# max_grad_norm: The maximum gradient norm.
# max_steps: The maximum number of steps to train the model for.
# warmup_ratio: The proportion of the training steps to use for warming up the learning rate.
# group_by_length: Whether to group the training samples by length.
# lr_scheduler_type: The type of learning rate scheduler to use.
# report_to: The tools to report the training metrics to.
# evaluation_strategy: The strategy for evaluating the model during training.
# SFTTrainer:
# 
# The SFTTrainer is a custom trainer class from the TRL library. It is used to train large language models (also using the PEFT method).
# 
# The SFTTrainer object is initialized with the following arguments:
# 
# model: The model to be trained.
# train_dataset: The training dataset.
# eval_dataset: The evaluation dataset.
# peft_config: The PEFT configuration.
# dataset_text_field: The name of the text field in the dataset.
# tokenizer: The tokenizer to use.
# args: The training arguments.
# packing: Whether to pack the training samples.
# max_seq_length: The maximum sequence length.
# Once the SFTTrainer object is initialized, it can be used to train the model by calling the train() method

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T02:40:51.065690Z","iopub.execute_input":"2024-08-01T02:40:51.066113Z","iopub.status.idle":"2024-08-01T06:22:30.816972Z","shell.execute_reply.started":"2024-08-01T02:40:51.066078Z","shell.execute_reply":"2024-08-01T06:22:30.815969Z"},"jupyter":{"outputs_hidden":false}}
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
)

training_arguments = TrainingArguments(
    output_dir="trained_weigths_4_EPOCH",
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=False,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    eval_steps=25,
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    packing=False,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False,
    },
)

trainer.train()
trainer.save_model()
tokenizer.save_pretrained("trained_weigths_4_EPOCH")

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T06:22:30.818188Z","iopub.execute_input":"2024-08-01T06:22:30.818811Z","iopub.status.idle":"2024-08-01T06:22:30.824770Z","shell.execute_reply.started":"2024-08-01T06:22:30.818776Z","shell.execute_reply":"2024-08-01T06:22:30.823790Z"},"jupyter":{"outputs_hidden":false}}
# Assuming X_eval has a 'sentiment' column with the true labels
y_true = X_eval["sentiment"].apply(lambda x: x.strip())

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T07:32:25.395613Z","iopub.execute_input":"2024-08-01T07:32:25.396026Z","iopub.status.idle":"2024-08-01T07:32:27.110203Z","shell.execute_reply.started":"2024-08-01T07:32:25.395992Z","shell.execute_reply":"2024-08-01T07:32:27.108996Z"},"jupyter":{"outputs_hidden":false}}
# y_pred = predict(eval_data.to_pandas(), model, tokenizer)
# y_true = X_eval["sentiment"].apply(lambda x: x.strip())
# evaluate(y_true, y_pred)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T06:22:31.888685Z","iopub.status.idle":"2024-08-01T06:22:31.889018Z","shell.execute_reply.started":"2024-08-01T06:22:31.888862Z","shell.execute_reply":"2024-08-01T06:22:31.888876Z"},"jupyter":{"outputs_hidden":false}}
print(X_eval)

# %% [markdown]
# The following code will train the model using the trainer.train() method and then save the trained model to the trained-model directory. Using The standard GPU P100 offered by Kaggle, the training should be quite fast.

# %% [markdown]
# The model and the tokenizer are saved to disk for later usage.

# %% [markdown]
# Afterwards, loading the TensorBoard extension and start TensorBoard, pointing to the logs/runs directory, which is assumed to contain the training logs and checkpoints for your model, will allow you to understand how the models fits during the training.

# %% [markdown]
# SETTING UP RAG

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T06:22:55.169999Z","iopub.execute_input":"2024-08-01T06:22:55.170365Z","iopub.status.idle":"2024-08-01T06:23:35.176476Z","shell.execute_reply.started":"2024-08-01T06:22:55.170332Z","shell.execute_reply":"2024-08-01T06:23:35.175587Z"},"jupyter":{"outputs_hidden":false}}
from datasets import Dataset, load_from_disk
from sentence_transformers import SentenceTransformer
import faiss

# Load passages (e.g., financial news articles)
passages = [text for text in df['text']]
titles = [f"Title {i}" for i in range(len(passages))]  # Dummy titles
embedding_model = SentenceTransformer('all-mpnet-base-v2')
passage_embeddings = embedding_model.encode(passages, convert_to_tensor=True)

# Create a Dataset object
dataset = Dataset.from_dict({"title": titles, "text": passages})
dataset = dataset.add_column("embeddings", [emb.tolist() for emb in passage_embeddings])

# Add a FAISS index
faiss_index = faiss.IndexFlatL2(passage_embeddings.shape[1])
dataset.add_faiss_index(column='embeddings', custom_index=faiss_index)

# Save the FAISS index separately
index_path = 'path_to_save_texts/passage_texts_index'
dataset.get_index("embeddings").save(index_path)

# Drop the FAISS index before saving the dataset
dataset.drop_index("embeddings")

# Save the dataset to disk
dataset_path = 'path_to_save_texts/passage_texts_dataset'
dataset.save_to_disk(dataset_path)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T06:23:39.860127Z","iopub.execute_input":"2024-08-01T06:23:39.860501Z","iopub.status.idle":"2024-08-01T06:23:39.923479Z","shell.execute_reply.started":"2024-08-01T06:23:39.860471Z","shell.execute_reply":"2024-08-01T06:23:39.922577Z"},"jupyter":{"outputs_hidden":false}}
# Load the dataset and re-add the FAISS index
dataset = load_from_disk(dataset_path)
faiss_index = faiss.read_index(index_path)
dataset.add_faiss_index(column='embeddings', custom_index=faiss_index)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T06:23:42.601215Z","iopub.execute_input":"2024-08-01T06:23:42.602093Z","iopub.status.idle":"2024-08-01T06:31:27.248240Z","shell.execute_reply.started":"2024-08-01T06:23:42.602058Z","shell.execute_reply":"2024-08-01T06:31:27.247234Z"},"jupyter":{"outputs_hidden":false}}
from transformers import RagRetriever, RagTokenizer, RagSequenceForGeneration

model_name = "facebook/rag-sequence-nq"
tokenizer = RagTokenizer.from_pretrained(model_name)
retriever = RagRetriever.from_pretrained(model_name, index_name="custom", passages_path=dataset_path, index_path=index_path)
model = RagSequenceForGeneration.from_pretrained(model_name, retriever=retriever).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T06:33:52.329630Z","iopub.execute_input":"2024-08-01T06:33:52.330545Z","iopub.status.idle":"2024-08-01T06:33:52.338252Z","shell.execute_reply.started":"2024-08-01T06:33:52.330510Z","shell.execute_reply":"2024-08-01T06:33:52.337292Z"},"jupyter":{"outputs_hidden":false}}
def predict_with_rag(test, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["text"]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Generate predictions using the RAG model
        generated = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        generated_text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        answer = generated_text.split("Sentiment:")[-1].strip()
        
        if "positive" in answer:
            y_pred.append("positive")
        elif "negative" in answer:
            y_pred.append("negative")
        else:
            y_pred.append("neutral")
    
    return y_pred

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T06:33:55.293647Z","iopub.execute_input":"2024-08-01T06:33:55.294512Z","iopub.status.idle":"2024-08-01T07:21:53.152597Z","shell.execute_reply.started":"2024-08-01T06:33:55.294479Z","shell.execute_reply":"2024-08-01T07:21:53.151655Z"},"jupyter":{"outputs_hidden":false}}
# Assuming X_eval has a 'sentiment' column with the true labels
y_true = X_eval["sentiment"].apply(lambda x: x.strip())
y_pred_rag = predict_with_rag(X_eval, model, tokenizer)

evaluate(y_true, y_pred_rag)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T07:29:49.067887Z","iopub.execute_input":"2024-08-01T07:29:49.068692Z","iopub.status.idle":"2024-08-01T07:29:51.481205Z","shell.execute_reply.started":"2024-08-01T07:29:49.068661Z","shell.execute_reply":"2024-08-01T07:29:51.480219Z"},"jupyter":{"outputs_hidden":false}}
def test_custom_document(custom_text, model, tokenizer):
    prompt = f"Text: '{custom_text}'\nSentiment: "
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    generated = model.generate(input_ids=input_ids, attention_mask=attention_mask)
    generated_text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    answer = generated_text.split("Sentiment:")[-1].strip()
    
    # Map the answer to one of the sentiment labels
    if "positive" in answer:
        return "positive"
    elif "negative" in answer:
        return "negative"
    else:
        return "neutral"

# Example usage
custom_text = "The company's stock price soared after the positive earnings report."
predicted_sentiment = test_custom_document(custom_text, model, tokenizer)
print(f"Predicted Sentiment: {predicted_sentiment}")
