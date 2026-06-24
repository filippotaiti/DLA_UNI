# Classification and Object Detection

Laboratorio sulla classificazione e sull'object detection, utilizzando dataset contenenti immagini stradali.

Il progetto si concentra inizialmente sulla classificazione, utilizzando sia una SVM che una ResNet50 e successivamente sull'object detection, utilizzando una Faster-R-CNN.

Il laboratorio è suddiviso in 3 esercizi: il primo e il terzo sono reperibili presso il seguente notebook: [`DLA-Lab1.ipynb`](./DLA-Lab1.ipynb), mentre il secondo è stato sviluppato in moduli Python separati, contenuti nella cartella ['src'](./src).

La struttura del secondo esercizio è la seguente:

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



In particolare, il secondo esercizio è stato sviluppato utilizzando il framework ['Hydra'](https://hydra.cc/docs/intro/), sviluppato da Meta.
L'obiettivo era quello di sviluppare una pipeline modulare, con un alto livello di astrazione, in maniera tale da essere il più possibile indipendente dal modello e dallo scenario generale.

La pipeline è composta da:

1. *config*: cartella che contiene a sua volta le cartelle *model* e *optimizer*, che contengono, rispettivamente, i file  resnet50.yaml e adam.yaml
2. *create_dls.py* : contiene la funzione che crea i DataLoaders
3. *factory_model.py*: contiene la funzione che istanzia la resnet
4. *logger.py*: implementa il sistema di logging con WandB
5. *train_and_evaluate.py*: implementa la pipeline di train+eval
6. *main.py*: modulo principale che può essere lanciato da terminale.

La pipeline è stata sviluppata attorno al primo esercizio, con l'obiettivo di riprodurre l'esercizio 1.3, che di base è stato svolto nel notebook.
Tuttavia, nonostante la pipeline sia 'ResNet-oriented', la configurazione sviluppata con Hydra è del tutto generica: modificando i file YAML (o creandone di nuovi) si possono istanziare anche altri modelli da usare per altri task (es. CLIPModel, ViT, ...)
    
Si può eseguire la pipeline scrivendo su terminale `python lab1/src/main.py`. 


## Implementazione:

### 1. Classification using a SVM (baseline) and a ResNet50

In questo esercizio, inizialmente viene svolta un'esplorazione dei dati (Exploration Data Analysis), sul dataset **GTSRB**. Successivamente si utilizza una ResNet50 pre-addestrata per fare *feature extraction*. 
Le feature estratte in questo modo, sono utilizzate per addestrare e valutare una SVM. Fino a questo punto, il pre-processing consiste soltanto nella conversione delle immagini in tensori.

La SVM ottiene sul test set un'accuratezza molto bassa: **68%**.

Successivamente, utilizzando un pre-processing più aggressivo (più adatto a una `ResNet50`), si va a fine-tunare una `ResNet50` pre-addestrata, ottenendo un'accuratezza del **95%**.

### 2. Pipeline consolidation

Riportato sopra

### 3. Detecting traffic signs

In quest'ultimo esercizio, si istanzia una `Faster-R-CNN', andando a sostituire il suo backbone con quello della ResNET 50 fine-tunata nell'esercizio 1.3 al fine di andare a valutare le prestazioni su **GTSRB Detection Dataset**.
Come metrica di valutazione viene utilizzata la mAP in varie forme:
- mAP
- mAP@50
- mAP@75
- mAP small
- mAP medium
- mAP large

---
 
## Risultati principali
 
| Esperimento | Metrica | Valore |
|---|---|---|
| Baseline SVM | Test accuracy | **68%** |
| Fine-tuned ResNet50 | Test accuracy| **95%** |
| Faster-R-CNN (ResNet50 backbone) | mAP | **0.5952** |
| Faster-R-CNN (ResNet50 backbone) | mAP@50 | **0.7277** |
| Faster-R-CNN (ResNet50 backbone) | mAP@75 | **0.7168** |
| Faster-R-CNN (ResNet50 backbone) | mAP small| **0.4660** |
| Faster-R-CNN (ResNet50 backbone) | mAP medium | **0.7334** |
| Faster-R-CNN (ResNet50 backbone) | mAP large | **0.75** |


----------

### Uso dell'AI:

1. Ho utilizzato Gemini per capire bene il funzionamento della faster_r_cnn. 
2. Nel secondo esercizio volevo cercare di generare un modello di pipeline riproducibile, non una semplice classe con WandB integrato, che fosse il più possibile indipendente dal modello, sia al fine di svolgere correttamente l'esercizio, ma anche di imparare qualcosa di nuovo che mi potrà tornare utile in futuro. 
In particolare, ho chiesto a Claude come strutturano le pipeline in aziende big tech e mi è stato suggerito questo framework Hydra, sviluppato da Meta, che io non conoscevo. Quindi mi sono fatto spiegare come funziona. Una volta imparato ciò l'ho adattato al mio caso, consultando anche la documentazione sotto-riportata.
