# Adversarial Learning & OOD Detection
 
Laboratorio sull'apprendimento avversario e il rilevamento di campioni *out-of-distribution* (OOD) su CIFR-10. Il progetto copre tre filoni collegati: una **pipeline di OOD detection** basata sulle uscite di un classificatore e su un autoencoder, l'implementazione del **Fast Gradient Sign Method (FGSM)** per attacchi avversari *targeted* e *untargeted*, e una **difesa** tramite *adversarial training*.
 
Tutto il codice è contenuto in un unico notebook: [`DLA-Lab4.ipynb`](./DLA-Lab4.ipynb).
 
---
 
## Indice
 
- [Panoramica](#panoramica)
- [Requisiti e installazione](#requisiti-e-installazione)
- [Struttura del progetto](#struttura-del-progetto)
- [Dataset e preprocessing](#dataset-e-preprocessing)
- [Metodi implementati](#metodi-implementati)
  - [1. OOD detection](#1-ood-detection)
  - [2. Attacchi avversari (FGSM)](#2-attacchi-avversari-fgsm)
  - [3. Adversarial training](#3-adversarial-training)
- [Risultati principali](#risultati-principali)
- [Come riprodurre](#come-riprodurre)
- [Riferimenti](#riferimenti)
---
 
## Panoramica
 
Il progetto si articola in tre esercizi che condividono lo stesso classificatore CNN e lo stesso setup di dati.
 
1. **OOD detection** - Si costruisce un punteggio di "normalità" per ogni immagine e si verifica quanto bene separa i dati *in-distribution* (CIFAR-10) da quelli *out-of-distribution* (rumore casuale generato da `FakeData`). Si confrontano due famiglie di detector: quelli basati sulle uscite del classificatore (max-logit e max-softmax) e quello basato sull'errore di ricostruzione di un autoencoder. La qualità della separazione è misurata con l'**AUROC**.
2. **Attacchi avversari** - Si implementa l'FGSM e lo si applica iterativamente per generare esempi avversari, sia *untargeted* (far sbagliare il modello in qualunque modo) sia *targeted* (forzare una classe specifica). Si misura l'efficacia con l'**Attack Success Rate (ASR)** e la **robust accuracy** al variare del budget di perturbazione.
3. **Difesa** - Si riaddestra il classificatore aumentando ogni batch con esempi avversari FGSM (*adversarial training*) e si verifica il miglioramento di robustezza.
---
 
## Requisiti e installazione
 
Il progetto è stato sviluppato con **Python 3.9** e gestito tramite [`uv`](https://github.com/astral-sh/uv).
 
Le dipendenze principali sono:
 
- `torch`, `torchvision` - modelli, dataset e trasformazioni
- `numpy` - manipolazione di array e ordinamenti
- `matplotlib` - grafici
- `scikit-learn` - metriche di valutazione (ROC, AUROC, matrice di confusione, accuratezza)
- `omegaconf` - gestione delle configurazioni
- `tqdm` - barre di avanzamento
### Setup con `uv`
 
```bash
# Crea l'ambiente virtuale
uv venv
 
# Attivalo
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
 
# Installa le dipendenze
uv pip install torch torchvision numpy matplotlib scikit-learn omegaconf tqdm
```
 
### Riproducibilità
 
Il notebook imposta i seed di `torch`, `numpy` e `random` e attiva `torch.use_deterministic_algorithms(True)`. Per questo motivo è impostata anche la variabile d'ambiente `CUBLAS_WORKSPACE_CONFIG=:4096:8` (richiesta da alcune operazioni deterministiche su CUDA).
 
---

 
Il dataset CIFAR-10 viene scaricato in automatico in `./data` la prima volta che si eseguono le celle di caricamento dati.
 
---
 
## Dataset e preprocessing
 
- **In-distribution (ID):** CIFAR-10 (10 classi, immagini 3×32×32).
- **Out-of-distribution (OOD):** `torchvision.datasets.FakeData`, che genera **immagini di rumore casuale**.
Le immagini sono trasformate con `torchvision.transforms.v2` e **normalizzate con `mean = (0.5, 0.5, 0.5)` e `std = (0.5, 0.5, 0.5)`**, portando i pixel nell'intervallo `[-1, 1]`. La stessa trasformazione è applicata a CIFAR-10 e a `FakeData`, così i punteggi dei detector sono confrontabili.
 
Una classe `NormalizeInverse` esegue la de-normalizzazione (`inv_normalize`), necessaria per visualizzare correttamente le immagini e per misurare le perturbazioni nello spazio dei pixel.
 
> **Budget di perturbazione e spazio normalizzato.** Poiché le immagini sono normalizzate, un valore `eps` (es. `5/255`) vincola la perturbazione L∞ *nello spazio normalizzato*. Con `std = 0.5`, la perturbazione L∞ corrispondente *nello spazio dei pixel* è `eps · 0.5` — ad esempio `5/255` in spazio normalizzato ≈ `2.5/255` sui pixel.
 
---
 
## Metodi implementati
 
### 1. OOD detection
 
Il modello di base è una **CNN** convoluzionale (5 layer convoluzionali + 3 fully-connected) addestrata alla classificazione su CIFAR-10 con `CrossEntropyLoss` e ottimizzatore Adam.
 
A partire dalle sue uscite si definiscono due punteggi di normalità (più alto = più probabilmente in-distribution):
 
- **`max_logit`** — il valore massimo dei logit grezzi. Non è limitato in un intervallo.
- **`max_softmax`** — la massima probabilità softmax (con parametro opzionale di *temperature*). È limitato in `[0, 1]`.
L'idea: un classificatore tende a essere più "sicuro" (uscita massima più alta) sui dati che somigliano a quelli di training, e meno sicuro sul rumore.
 
In parallelo si addestra un **autoencoder** convoluzionale (encoder/decoder simmetrici, attivazione finale `Tanh` coerente con il range `[-1, 1]`) **solo sulle immagini in-distribution**. Il punteggio di normalità è l'**errore di ricostruzione MSE cambiato di segno** (`-MSE`): le immagini ben ricostruite (basso errore) ottengono un punteggio alto, quelle ricostruite male (alto errore, tipiche dei dati OOD) un punteggio basso.
 
La performance di entrambi i detector è valutata con la **curva ROC** e l'**AUROC**, metriche *threshold-free* (non richiedono di fissare una soglia).
 
### 2. Attacchi avversari (FGSM)
 
La funzione `FGSM` implementa il passo base del Fast Gradient Sign Method:
 
- calcola il gradiente della loss rispetto all'immagine di input;
- costruisce la perturbazione `eps · sign(gradiente)`;
- per l'attacco **untargeted** somma la perturbazione (`x + eps·sign`), allontanando l'immagine dalla classe corretta;
- per l'attacco **targeted** la sottrae (`x - eps·sign`), avvicinando l'immagine a una classe bersaglio.
Le funzioni `untargeted_attack` e `targeted_attack` applicano l'FGSM **iterativamente** (in stile *Basic Iterative Method*): partono dall'immagine pulita e accumulano passi di ampiezza `eps` finché l'attacco riesce o si raggiunge un numero massimo di iterazioni, riportando il *budget* L∞ totale impiegato.
 
### 3. Adversarial training
 
La difesa riaddestra il classificatore con una variante dell'adversarial training: per ogni batch si generano esempi avversari FGSM al volo e si calcola la loss su di essi, in modo che il modello impari a classificare correttamente anche gli input perturbati.
 
La robustezza del modello difeso è valutata con un attacco **targeted** ripetuto su tutte le classi bersaglio, misurando ASR e robust accuracy. L'analisi è ripetuta per **diversi valori di budget** (`eps` da `3/255` a `8/255`) per tracciare le curve di ASR e robust accuracy in funzione della perturbazione.
 
#### Metriche
 
- **Attack Success Rate (ASR)** — frazione di campioni *attaccabili* che l'attacco riesce a ingannare. Per l'attacco targeted, "attaccabile" esclude i campioni la cui classe vera è già il bersaglio e quelli già predetti come bersaglio.
- **Robust accuracy** — accuratezza del modello *sugli esempi avversari*. È sempre ≤ clean accuracy (un attacco può solo rompere predizioni corrette, non ripararne).
- **Clean accuracy** — accuratezza sulle immagini pulite, riportata come riferimento.
---
 
## Risultati principali
 
| Esperimento | Metrica | Valore |
|---|---|---|
| Classificatore CNN (base) | Test accuracy (CIFAR-10) | **69.37%** |
| Classificatore CNN dopo adversarial training | Test accuracy (CIFAR-10) | **70.69%** |
| OOD detection — Autoencoder (`-MSE`) | AUROC (CIFAR-10 vs FakeData) | **1.00** |
| OOD detection — CNN (`max_logit`) | AUROC (CIFAR-10 vs FakeData) | **0.92** |
 
**Osservazioni:**
 
- **OOD detection.** L'autoencoder separa perfettamente immagini reali e rumore (AUROC 1.00): non avendo mai visto rumore in training, lo ricostruisce malissimo, producendo un errore nettamente più alto. La CNN ottiene un AUROC più basso (0.92) perché le reti possono classificare il rumore con confidenza elevata, producendo talvolta un max-logit alto che ne riduce il potere discriminante. **Importante:** `FakeData` è rumore puro, quindi questo è il caso *facile* di OOD detection — l'AUROC alto va letto come "in-distribution vs rumore casuale", non come robustezza contro anomalie difficili (es. esempi avversari).
- **Attacchi.** Gli attacchi *targeted* richiedono in generale un budget maggiore di quelli *untargeted*, perché devono raggiungere una classe specifica anziché una misclassificazione qualsiasi. All'aumentare di `eps`, l'ASR cresce e la robust accuracy cala in modo monotono.
- **Difesa.** L'adversarial training migliora la test accuracy di circa **2 punti percentuali** rispetto al modello base.
> I grafici (curve ROC, istogrammi dei punteggi, esempi di immagini avversarie con perturbazione amplificata, curve ASR/robust accuracy vs `eps`) sono generati ed embeddati direttamente nel notebook.
 
---
 
## Come riprodurre
 
Il notebook è pensato per essere eseguito **in ordine, dall'alto verso il basso**. La sequenza logica è:
 
1. **Setup** - import, seed, configurazione, trasformazioni e DataLoader (CIFAR-10 + FakeData). CIFAR-10 viene scaricato automaticamente.
2. **Classificatore** -definizione e addestramento della CNN, valutazione (accuratezza + matrice di confusione).
3. **OOD detection con la CNN** - calcolo dei punteggi `max_logit`/`max_softmax`, grafici delle distribuzioni e curva ROC.
4. **OOD detection con l'autoencoder** - addestramento dell'autoencoder, calcolo dei punteggi di ricostruzione, grafici e curva ROC.
5. **Attacchi FGSM** - definizione di `FGSM` e delle funzioni di attacco; esempi di attacco untargeted e targeted.
6. **Adversarial training** - riaddestramento del classificatore con esempi avversari e nuova valutazione.
7. **Valutazione targeted multi-budget** - `targeted_FGSM_with_eval` ed `evaluation` su tutte le classi e per diversi `eps`; curve finali ASR/robust accuracy vs budget.


------------

## Uso dell'AI ##

Ho utilizzato Claude nell'esercizio 3.3 nei seguenti scenari:

1. Aiutarmi a generare dei grafici che evidenziassero i risultati ottenuti, in maniera tale da avere una buona visualizzazione. In particolare, non conoscevo l'esistenza del metodo *axline()* per disegnare delle linee su dei grafici ed ora si grazie all'utilizzo dell'IA.
2. Volevo rappresentare la perturbazione data dagli attacchi, ma l'immagine risultante era tutta nera e non capivo come mai. Claude mi ha chiarito la motivazione e mi ha suggerito come modificare per fare in modo che si veda.
3. Per aiutarmi a generare un README.md ben fatto.


