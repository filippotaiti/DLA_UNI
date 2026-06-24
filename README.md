# DLA_UNI #

This repository contains the lab exercises for the Deep Learning Applications course. 
The selected labs are:
- From Pixels to Semantics
- The Transformative Transformer
- OOD Detection and Adversarial Robustness.

[uv](https://docs.astral.sh/uv/pip/environments/) has always been the only tool used to generate virtual environments.

---

## Environment setup ##

To run notebooks and script, simply clone the repository and use *uv* to synchronize the environment.

---

1. ### Clone the repository:

```bash
git clone https://github.com/filippotaiti/DLA_UNI.git
cd DLA_UNI
```

2. ### Create the virtual environment and install dependencies: 


```bash
uv sync
```

This command creates a virtual environment and installs all the necessary packages.

---

## Repo structure ##
The repository is organized into three subfolders, one for each lab:
1. Lab 1: From Pixels to Semantics: 
- `DLA-Lab1.ipynb`: A notebook containing the solutions to the first and third exercises.
- `src`: this folder contains the second exercise of the lab.

2. Lab 2: The Transformative Transformer
- `DLA-Lab2.ipynb`: A notebook containing the solutions to all the exercises.

3. Lab 4: OOD Detection and Adversarial Robustness
- `DLA-Lab4.ipynb`: A notebook containing the solutions to all the exercises.

Each of the three subfolders has a dedicated README.md file that explains the exercise in detail.
Specifically, the README.md file for the first lab explains how to run the script for the second exercise.

---

### How to execute ###

After the execution of `uv sync`, the notebooks can be opened and run in an IDE (e.g., VS Code, PyCharm, ...). Be sure, however, to select the correct Python kernel.
Alternatively, notebooks can be run directly from the terminal using the following command:

```bash
cd labX
uv run jupyter nbconvert --to notebook --execute --inplace DLA-LabX.ipynb
```
