# Lab 1: From Pixels to Semantics

Lab on classification and object detection, using datasets containing street images.

The project initially focuses on classification, using both an **SVM** and a **ResNet50**, and subsequently on object detection, using a **Faster-R-CNN**.

The lab is divided into 3 exercises: the first and third can be found in the following notebook: [`DLA-Lab1.ipynb`](./DLA-Lab1.ipynb), while the second was developed in separate Python modules, located in the folder [src](./src).

The structure of the second exercise is as follows:

```text
src
├── config
│   ├── model
│   │   └── resnet50.yaml
│   ├── optimizer
│   │   └── adam.yaml
│   └── train_file.yaml
├── create_dls.py
├── factory_model.py
├── logger.py
├── main.py
└── train_and_evaluate.py
```

Specifically, the second exercise was developed using the framework  [Hydra](https://hydra.cc/docs/intro/).
The goal was to develop a modular pipeline with a high level of abstraction, so that it would be as independent as possible from the model and the overall scenario.

The pipeline consists of:

1. *config*: a folder that contains the *model* and *optimizer* folders, which contain the *resnet50.yaml* and *adam.yaml* files, respectively
2. *create_dls.py* : contains the function that creates the DataLoaders
3. *factory_model.py*: contains the function that instantiates the ResNet
4. *logger.py*: implement the logging system using **WandB**
5. *train_and_evaluate.py*: Implements the training-and-evaluation pipeline
6. *main.py*: main module that can be launched from the terminal

The pipeline was developed around the first exercise, with the goal of replicating Exercise 1.3, which was originally carried out in the notebook.
However, although the pipeline is “ResNet-oriented,” the configuration developed with Hydra is entirely generic: by modifying the YAML files (or creating new ones), you can also instantiate other models for use in different tasks (e.g., CLIPModel, ViT, ...).
    
You can run the pipeline using the following command: 

```bash
python lab1/src/main.py
```


## Implementation:

### 1. Classification using a SVM (baseline) and a ResNet50

In this exercise, we first perform an exploration of the data (Exploratory Data Analysis) on the **GTSRB** dataset. Next, we use a pre-trained ResNet50 to perform *feature extraction*.
The features extracted in this way are used to train and evaluate an SVM. Up to this point, preprocessing consists solely of converting the images into tensors.

The SVM achieves a low accuracy on the test set: **68%**.

Next, using more aggressive preprocessing (better suited for a pre-trained `ResNet50`), we fine-tune a pre-trained `ResNet50`, achieving an accuracy of **95%**.

### 2. Pipeline Consolidation

Described above.

### 3. Detecting Traffic Signs
In this final exercise, we instantiate a `Faster-R-CNN`, replacing its backbone with that of the ResNet 50 fine-tuned in Exercise 1.3 in order to evaluate its performance on the **GTSRB Detection Dataset**.
We use mAP in various forms as the evaluation metric:
- mAP
- mAP@50
- mAP@75
- mAP small
- mAP medium
- mAP large

---
 
## Main results
 
| Experiment | Metric | Value |
|---|---|---|
| Baseline SVM | Test accuracy | **68%** |
| Fine-tuned ResNet50 | Test accuracy| **95%** |
| Faster-R-CNN (ResNet50 backbone) | mAP | **0.5952** |
| Faster-R-CNN (ResNet50 backbone) | mAP@50 | **0.7277** |
| Faster-R-CNN (ResNet50 backbone) | mAP@75 | **0.7168** |
| Faster-R-CNN (ResNet50 backbone) | mAP small| **0.4660** |
| Faster-R-CNN (ResNet50 backbone) | mAP medium | **0.7334** |
| Faster-R-CNN (ResNet50 backbone) | mAP large | **0.75** |

## How to play
 
The notebook is designed to be worked through **in order, from top to bottom**.


----

### AI use:

1. I used **Gemini** to fully understand how `faster_r_cnn` works. 
2. In the second exercise, I wanted to try to create a reproducible pipeline model - not just a simple class with a built-in logging system - that was as model-independent as possible, both to complete the exercise correctly and to learn something new that might come in handy in the future. 
Specifically, I asked **Claude** how they structure pipelines at big tech companies, and he suggested this Hydra framework, which I wasn’t familiar with. So I had him explain how it works. Once I understood it, I adapted it to my situation, also referring to the documentation.
