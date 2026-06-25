# Lab 2: The Transformative Transformer
 
Lab on transformers and CLIP. 

The project covers two aspects: **sentiment analysis** and **fine-tuning a CLIP model**.
 
All the code is contained in a single notebook: [`DLA-Lab2.ipynb`](./DLA-Lab2.ipynb).
 
## Overview

The project consists of three exercises:
 
1. **Sentiment Analysis**: an **exploratory data analysis** is performed on the dataset, and then a pre-trained **DistilBERT** model is used for *feature extraction*. The output of the *feature extraction* is then used to train an SVM, in order to establish a baseline.
2. **Fine-tuning of DistilBERT**: DistilBERT is being fine-tuned in the hope of achieving better results than those obtained in the previous exercise.
3. **Fine-tuning of a CLIP Model**: given the [`openai/clip-vit-base-patch16`](https://huggingface.co/openai/clip-vit-base-patch16) model, we evaluate its performance on the ImageNette dataset. 
Next, we evaluate its zero-shot accuracy on a sketch dataset and then, in order to improve it, we fine-tune the model. The fine-tuning was performed using the **LoRA** technique.

 
## Dataset used
 
- **Sentiment Analysis & Fine-tuning DistilBERT**: [Cornell Rotten Tomatoes movie review dataset](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)
- **Fine-tuning of a CLIP Model**: ImageNette, [Sketchy Image Dataset] (https://www.kaggle.com/datasets/dhananjayapaliwal/fulldataset)
 
---
 
## Implementation
 
### 1. SVM (baseline)
The baseline model is an **SVM** with `kernel = linear`. It is trained and evaluated on features extracted from DistilBERT.
 
### 2. Fine-Tuning of DistilBERT
DistilBERT is fine-tuned **on the same dataset from which the features were previously extracted** in order to improve its performance.
 
### 3. Fine-Tuning of a CLIP Model
In this exercise, we first evaluate the zero-shot accuracy of the **CLIP model** on ImageNette, achieving a flawless result (**99%**). Then, to test the model's capabilities, we use a dataset of sketch images.

In this dataset, the zero-shot accuracy is very poor. However, it can be improved through fine-tuning using a **PEFT** technique, namely LoRA.

LoRA was applied only to the attention layers, specifically to the query and value components. This results in training with fewer than 500k parameters (instead of the default approximately 150M).

The fine-tuning led to a significant improvement in accuracy.

---
 
## Main results
 
| Experiment | Metric | Value |
|---|---|---|
| Baseline SVM | Test accuracy | **81%** |
| Fine-tuned DistilBert | Test accuracy| **83.7%** |
| CLIP Model BEFORE Fine-Tuning | Test accuracy | **65.56%** |
| CLIP Model AFTER Fine-Tuning | Test accuracy | **88.6%** |
 

## How to play
 
The notebook is designed to be worked through **in order, from top to bottom**.

------------


### AI Use:

I used **Claude** to help me understand the HuggingFace documentation (LoRA/PEFT) and to figure out how to set up OmegaConf correctly, in order to avoid using a simple Python dictionary or a .YAML file, since I don’t consider accessing them that way (i.e., `dict[key] = value`) to be aesthetically pleasing or easy to read for someone outside the project who needs to understand the code.