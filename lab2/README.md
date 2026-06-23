# Transformers and CLIP
 
Laboratorio sui transformers e su CLIP. 

Il progetto copre due aspetti: **sentiment analysis**  e **fine-tuning di un CLIP Model**.
 
Tutto il codice è contenuto in un unico notebook: [`DLA-Lab2.ipynb`](./DLA-Lab2.ipynb).
 
---
 
## Indice
 
- [Panoramica](#panoramica)
- [Dataset utilizzati](#dataset-utilizzati)
- [Metodi implementati](#metodi-implementati)
  - [1. SVM (baseline)](#1-svm-(baseline))
  - [2. Fine-Tuning of DistilBERT](#2-fine-tuning-of-distilbert)
  - [3. Fine-Tuning of a CLIP Model](#3-fine-tuning-of-a-clip-model)
- [Risultati principali](#risultati-principali)
- [Come riprodurre](#come-riprodurre)
---
 
## Panoramica
 
Il progetto si articola in tre esercizi:
 
1. **Sentiment Analysis** - Si fa una **exploratory data analysis** del dataset utilizzato, successivamente si utilizza **DistilBERT** pre-addestrato per fare *feature extraction*. Con l'output della *feature extraction* si va ad addestrare una SVM, in modo da avere una baseline.
2. **Fine-tuning of DistilBERT** - Si fa fine-tuning di DistilBERT nella speranza di ottenere risultati migliori di quelli ottenuti nell'esercizio precedente.
3. **Fine-tuning of a CLIP Model**: dato il modello [`openai/clip-vit-base-patch16`](https://huggingface.co/openai/clip-vit-base-patch16), si valutano le sue prestazioni sul dataset ImageNette. 
Poi si valuta l'accuratezza zero-shot su un dataset di sketch e successivamente, al fine di migliorarla, se ne fa il fine-tuning. Il fine-tuning è stato effettuato utilizzando la tecnica LoRA.

 
## Dataset utilizzati
 
- **Sentiment Analysis & Fine-tuning DistilBERT**: [Cornell Rotten Tomatoes movie review dataset](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)
- **Fine-tuning of a CLIP Model**: ImageNette, [Sketchy Image Dataset] (https://www.kaggle.com/datasets/dhananjayapaliwal/fulldataset)
 
---
 
## Metodi implementati
 
### 1. SVM (baseline)
 
Il modello che funge da baseline è una **SVM** con `kernel = linear`. Essa viene addestrata e valutata sulle feature estratte da DistilBERT.
 
### 2. Fine-Tuning of DistilBERT
 
Viene fatto il fine-tuning di DistilBERT **sullo stesso dataset da cui sono state estratte le feature precedentemente** al fine di migliorarne le prestazioni.
 
### 3. Fine-Tuning of a CLIP Model

In questo esercizio inizialmente si valuta l'accuratezza zero-shot del **modello CLIP** su ImageNette, ottenendo un risultato impeccabile (**99%**). Poi, al fine di mettere alla prova le capacità del modello, si utilizza un dataset di immagini sketch.
In tale dataset, l'accuratezza zero-shot è pessima. Tuttavia, la si va a migliorare effettuando fine-tuning, utilizzando una tecnica **PEFT**, cioè LoRA.

LoRA è stata applicata solo sui layer di attenzione, limitatamente a query e value. Questo comporta un addestramento di meno di 500k parametri (anziché i 150M circa di default).

Il fine-tuning ha portato ad un miglioramento notevole in termini di accuracy.
---
 
## Risultati principali
 
| Esperimento | Metrica | Valore |
|---|---|---|
| Baseline SVM | Test accuracy | **81%** |
| Fine-tuned DistilBert | Test accuracy| **83.7%** |
| CLIP Model BEFORE Fine-Tuning | Test accuracy | **65.56%** |
| CLIP Model AFTER Fine-Tuning | Test accuracy | **88.3%** |
 
**Osservazioni:**
 
 
## Come riprodurre
 
Il notebook è pensato per essere eseguito **in ordine, dall'alto verso il basso**.

------------


### Uso dell'AI:

Ho utilizzato Claude per aiutarmi a capire la documentazione di HuggingFace (LoRA/PEFT) e per capire come fare il setup in maniera corretta di OmegaConf, al fine di evitare di utilizzare un semplice dizionario Python, o comunque un file .YAML, dato che l'accesso ad essi (cioè dict[key] = value) non lo reputo esteticamente bello da vedere e da leggere per una persona esterna che deve capire il codice.