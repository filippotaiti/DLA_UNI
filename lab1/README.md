# Lab 1: Object Detection

In questo laboratorio, oltre ad aver svolto il primo esercizio, ho deciso di svolgere il secondo esercizio, che richiedeva di sviluppare una pipeline con logging per addestrare i modelli, utilizzando il framework Hydra, sviluppato da Meta, in modo da rendere la pipeline il meno hard-coded possibile. Dovendo poi testarne il corretto funzionamento, ho deciso di scriverla in 'stile' es 1.3 (in particolare il file *train_and_evaluate.py*).
Quindi, la pipeline è in parte "ResNet oriented", tuttavia la struttura della configurazione con Hydra è del tutto generale: modificando i file YAML (o creandone di nuovi), si possono istanziare anche altri modelli (e.g. CLIPModel, ViT, ...).

La pipeline si trova nella cartella *src*.

*src* contiene:

1. *config*: cartella che contiene a sua volta le cartelle *model* e *optimizer*, che contengono, rispettivamente, i file  resnet50.yaml e adam.yaml
2. *create_dls.py* : contiene la funzione che crea i DataLoaders
3. *factory_model.py*: contiene la funzione che istanzia la resnet
4. *logger.py*: implementa il sistema di logging con WandB
5. *train_and_evaluate.py*: implementa la pipeline di train+eval
6. *main.py*: modulo principale che può essere lanciato da terminale.


Si può eseguire la pipeline scrivendo su terminale " python lab1/src/main.py " 


Uso dell'AI:

1. Ho utilizzato Gemini per capire bene il funzionamento della faster_r_cnn. 
2. Nel secondo esercizio volevo cercare di generare un modello di pipeline riproducibile, non una semplice classe con WandB integrato, che fosse il più possibile indipendente dal modello, sia al fine di svolgere correttamente l'esercizio, ma anche di imparare qualcosa di nuovo che mi potrà tornare utile in futuro in una realtà lavorativa. 
In particolare, ho chiesto a Claude come strutturano le pipeline in aziende big tech e mi è stato suggerito questo framework Hydra, sviluppato da Meta, che io non conoscevo. Quindi mi sono fatto spiegare come funziona. Una volta imparato ciò l'ho adattato al mio caso in esame.