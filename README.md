# Multilabel FAERS Serious Outcome Prediction with a Heterogeneous GNN

This project uses a heterogeneous graph neural network to predict serious outcomes from FDA Adverse Event Reporting System (FAERS) reports.

The model combines patient and drug information from FAERS with biological information from BioDecagon and PubChem. Each patient is treated as a multilabel classification problem because a single FAERS report can contain several serious outcomes at the same time.

## Outcomes

The model predicts seven serious outcome categories:

| Code | Outcome |
|---|---|
| `DE` | Death |
| `LT` | Life-threatening |
| `HO` | Hospitalisation |
| `DS` | Disability |
| `CA` | Congenital anomaly |
| `RI` | Required intervention to prevent permanent impairment/damage |
| `OT` | Other serious outcome |

A patient can therefore have several positive labels rather than being forced into one class.

---

## Graph Structure

The graph contains:

- patients
- drugs
- chemicals
- genes
- indication terms
- disease classes

Main relationships include:

```text
patient -> drug
drug -> chemical
drug -> indication
chemical -> gene
gene -> gene
chemical -> chemical
chemical -> disease class
```

The final prediction is made at the patient node.

---

## Data Sources

The project combines:

- **FAERS** for patient reports, drug exposure, indications and serious outcomes
- **BioDecagon** for drug side effects, chemical-gene targets and protein-protein interactions
- **PubChem** for chemical identifiers and SMILES strings
- **BioBERT** for biomedical text embeddings
- **ChemBERTa** for SMILES embeddings

The raw datasets are not included in this repository.

---

## Preprocessing Pipeline

A simplified version of the pipeline is:

```text
FAERS quarterly files
        |
        v
data_extract.py
        |
        v
drug_therapy_cleaner.py
        |
        v
demo_merger_multilabel_outcomes.py
        |
        v
whole_db_merger.py
        |
        v
demo_cleaner.py
        |
        v
stitch_id_attacher.py
        |
        v
smiles_attacher.py
        |
        v
gene_list.py / side_effect_list.py
        |
        v
pre_embedder.py
        |
        v
graph_creator_outcome_v2.py
        |
        v
graph_data.pt
```

### Multilabel targets

The FAERS outcome table can contain multiple outcomes for the same report.

The current pipeline converts those rows into seven binary target columns:

```python
['DE', 'LT', 'HO', 'DS', 'CA', 'RI', 'OT']
```

The final patient target tensor has shape:

```text
[number_of_patients, 7]
```

This avoids losing information from reports with several serious outcomes.

---

## Node and Edge Features

### Patient nodes

Categorical features:

- manufacturer / sender
- occurrence country
- sex

Numerical features:

- event year
- age

### Patient-to-drug edges

Features include:

- drug role
- route of administration
- dechallenge
- rechallenge
- dose unit
- dose amount
- treatment duration
- missing-dose indicator

### Drug nodes

Drug names are represented using BioBERT embeddings.

### Chemical nodes

Chemical features combine:

- BioBERT chemical-name embeddings
- ChemBERTa SMILES embeddings
- mean embeddings of known mono-drug side effects

### Gene nodes

Genes use learned embeddings and are connected using BioDecagon protein-protein interactions.

---

## Model

The main training script is:

```text
GNN_multilabel_outcome_training.py
```

The model is implemented with PyTorch Geometric and uses relation-specific message passing across the heterogeneous graph.

Main settings:

```text
Hidden size:        128
GNN layers:         3
Neighbour samples:  [10, 7, 5]
Dropout:            0.2
Optimiser:          AdamW
Learning rate:      1e-3
Weight decay:       1e-4
Batch size:         256
```

The architecture uses:

- `SAGEConv` for most graph relations
- `NNConv` where edge features are important
- residual connections
- learned embeddings for the seven outcome labels

The output is:

```text
[batch_size, 7]
```

logits.

Because the targets are multilabel, training uses:

```python
BCEWithLogitsLoss
```

rather than softmax cross-entropy.

---

## Leakage Prevention

Patients can share drug nodes, so allowing messages to travel:

```text
patient A -> drug -> patient B
```

could leak patient information across examples.

To avoid this, the forward:

```text
patient -> drug
```

message-passing relation is removed during training.

The reverse:

```text
drug -> patient
```

relation is retained so patients can still receive information from their drugs.

---

## Class Imbalance

Approximate positive counts in the full dataset:

| Outcome | Positive reports |
|---|---:|
| DE | 104,974 |
| LT | 75,196 |
| HO | 407,783 |
| DS | 25,848 |
| CA | 6,450 |
| RI | 1,388 |
| OT | 549,114 |

Positive class weights are calculated from the training split using:

```python
sqrt(negative_count / positive_count)
```

and capped at `25`.

---

## Training and Evaluation

The current split is:

```text
80% training
10% validation
10% test
```

with seed `42`.

Validation **Macro AUPRC** is used for early stopping.

After training, a separate decision threshold is tuned for each outcome using the validation precision-recall curve. These thresholds are then fixed and applied to the test set.

---

## Results

### Overall test performance

| Metric | Score |
|---|---:|
| Macro F1 | 0.5273 |
| Micro F1 | 0.7152 |
| Weighted F1 | 0.7279 |
| Macro Precision | 0.4636 |
| Macro Recall | 0.6314 |
| Macro AUROC | 0.8596 |
| Macro AUPRC | 0.5158 |

### Per-outcome performance

| Outcome | Precision | Recall | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| DE | 0.4822 | 0.5293 | 0.5046 | 0.8516 | 0.5195 |
| LT | 0.1938 | 0.4612 | 0.2730 | 0.7451 | 0.2150 |
| HO | 0.6573 | 0.9232 | 0.7679 | 0.8121 | 0.7763 |
| DS | 0.2646 | 0.2578 | 0.2612 | 0.8065 | 0.2225 |
| CA | 0.5955 | 0.8167 | 0.6888 | 0.9754 | 0.6674 |
| RI | 0.2812 | 0.5436 | 0.3707 | 0.9947 | 0.2979 |
| OT | 0.7704 | 0.8876 | 0.8248 | 0.8317 | 0.9122 |

`LT` and `DS` are currently the weakest outcomes.

`DE` has reasonable ranking performance but only moderate recall, so a substantial number of true death reports are still missed.

For rare outcomes such as `RI`, AUPRC is more informative than AUROC alone.

---

## Baselines and Sanity Checks

`GNN_sanity_checks_and_baselines.py` compares the full graph against simpler models and a shuffled-label control.

| Model | Macro F1 | Macro AUROC | Macro AUPRC |
|---|---:|---:|---:|
| Patient/report only | 0.4424 | 0.8151 | 0.4045 |
| Patient/report + drugs | 0.5110 | 0.8547 | 0.4998 |
| Full heterogeneous GNN | **0.5273** | **0.8596** | **0.5158** |
| Shuffled-label GNN | 0.2736 | 0.4999 | 0.2003 |

The shuffled-label model fell to approximately random performance:

```text
Macro AUROC = 0.4999
```

and each label's AUPRC fell to approximately its prevalence.

This provides a useful sanity check that the model is not simply exploiting a direct target leakage bug.

The comparison also shows that drug identity already contains a large amount of predictive signal. The larger biological graph gives a smaller overall improvement, although its contribution varies by outcome.

For `DE`, the full graph provides a more noticeable improvement than the overall average.

---

## Single-Patient Inference

For a patient already present in the graph:

```bash
python single_patient_graph_creator.py
python single_patient_graph_tester.py
```

The tester prints each outcome's:

- predicted probability
- tuned threshold
- predicted label
- actual label

This is useful for inspecting individual model behaviour.

---

## External 2026 FAERS Test

`external_2026q2_patient_graph_creator.py` supports a simple temporal inference test on 2026 Q2 FAERS reports.

The script:

1. reads 2026 Q2 FAERS files
2. finds a compatible patient
3. requires the patient's drugs to already exist in the trained graph
4. appends the patient to the existing graph
5. connects the patient to the existing drug nodes
6. samples the same biological neighbourhood used during training
7. saves the resulting patient subgraph

Run:

```bash
python external_2026q2_patient_graph_creator.py
python single_patient_graph_tester.py
```

The creator can be restricted to a particular outcome:

```python
REQUIRE_OUTCOME = 'DE'
```

or set to:

```python
REQUIRE_OUTCOME = None
```

to select any compatible serious report.

A single external patient is useful for debugging, but a proper temporal evaluation should use a full 2026 cohort.

---

## Main Dependencies

```text
pandas
numpy
torch
torch-geometric
scikit-learn
transformers
tqdm
requests
python-dateutil
```

CUDA is strongly recommended for training.

---

## Limitations

### FAERS is observational

FAERS reports do not prove that a drug caused an outcome.

The model predicts patterns associated with **reported serious outcomes**, not causal drug risk.

### Reporting-context features

Manufacturer/sender and occurrence country are currently used as patient features.

These may capture reporting behaviour rather than biological mechanisms.

### Random patient split

The current train/test split is random at the patient level.

Drugs, chemicals and genes remain shared across the global graph, so the experiment tests new patients involving known graph entities rather than completely unseen drugs.

### Follow-up versions of cases

FAERS can contain several `primaryid` versions of the same `caseid`.

A grouped split by `caseid` would be a stronger evaluation and would prevent versions of the same case appearing across train and test sets.

### Rare outcomes

`CA` and especially `RI` have relatively few positive examples, so their metrics can change substantially from a small number of predictions.

### Drug normalisation

Drug synonym and PubChem mapping are still partly heuristic, and some drug names can map ambiguously to several identifiers.

---

## Future Work

### Grouped case split

Split the dataset by `caseid` so all follow-up versions of the same FAERS case remain in the same partition.

### Full temporal evaluation

Use:

```text
Train: 2015-2025
Test:  2026
```

to evaluate future-report generalisation rather than testing individual 2026 cases.

### Remove reporting-context features

Retrain without:

- manufacturer / sender
- occurrence country

to measure how much performance comes from pharmacological and biological information rather than reporting patterns.

### Drug-disjoint evaluation

Hold out complete drugs during training to test whether the biological graph can generalise to unseen treatments.

### Improve DE and LT prediction

Possible additions include:

- number of simultaneous drugs
- patient-level indication summaries
- comorbidity information
- explicit polypharmacy interaction features
- improved temporal features around treatment and event dates

### Better drug normalisation

Replace the current synonym matching with more structured ingredient-level mappings, for example using RxNorm-style normalisation.

### Explicit unknown categories

Create an `UNKNOWN` category during preprocessing for every categorical feature so unseen external reports can be handled consistently.

### Calibration

The sigmoid outputs are useful ranking scores but are not calibrated clinical probabilities.

Future work could test:

- temperature scaling
- isotonic regression
- calibration curves
- expected calibration error

### Explainability

Investigate which drugs and graph neighbours contribute most to an individual prediction using:

- drug ablation
- edge ablation
- integrated gradients
- PyTorch Geometric explainers

This would be useful for analysing false negatives, particularly for `DE` and `LT`.

### Hyperparameter tuning

Once the evaluation setup is finalised, useful parameters to explore include:

- hidden dimension
- number of GNN layers
- neighbour sample sizes
- dropout
- learning rate
- relation-specific aggregators
- class-weight transformations

---

## Current Status

The current model shows that serious FAERS outcomes can be predicted from a heterogeneous pharmacovigilance graph.

The main findings so far are:

- patient/report features already contain substantial predictive signal
- drug identity adds a large amount of information
- the biological graph gives a smaller but measurable improvement overall
- graph contribution varies considerably between outcomes
- shuffled-label training falls to random performance
- rare and heterogeneous outcomes remain harder to predict

The next priority is improving the evaluation setup rather than simply increasing model size: grouped case splits, a complete 2026 temporal test and stricter baselines should provide a better measure of how useful the biological graph actually is.
