from __future__ import annotations

import copy
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, NNConv, SAGEConv
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

SEED = 42

NUM_SELECTED_PTS = 1000
NUM_HARD_NEGATIVES = 50

TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10

HIDDEN_CHANNELS = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 10

TRAIN_BATCH_SIZE = 400
EVAL_BATCH_SIZE = 400

TRAIN_NUM_NEIGHBORS = [10, 7, 5, 3]
EVAL_NUM_NEIGHBORS = [10, 7, 5, 3]

RECALL_K_VALUES = (10, 50)

CHECKPOINT_TO_LOAD: Optional[Path] = None
OUTPUT_CHECKPOINT = Path("gnn_patient_multilabel_top1000_hard50.pth")


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# GNN model
# ============================================================

class AdvGNNPatientClassifier(nn.Module):
    """
    Patient-level multilabel classifier.

    The heterogeneous GNN creates a patient representation using the patient's
    attributes, drugs, chemicals, genes, indication terms, and other graph
    context.

    A single linear layer then produces one score for every selected PT label:

        patient representation -> [num_selected_pts logits]

    Patient-to-PT edges must not be present in the graph passed into forward(),
    because those edges are the prediction targets.
    """

    def __init__(
        self,
        hidden_channels: int,
        patient_feat_dim: int,
        chemical_feat_dim: int,
        demo_edge_feat_dim: int,
        num_genes: int,
        num_selected_pts: int,
        side_effect_emb: nn.Embedding,
        rept_emb: nn.Embedding,
        mfr_emb: nn.Embedding,
        occr_emb: nn.Embedding,
        sex_emb: nn.Embedding,
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_selected_pts = num_selected_pts

        # ----------------------------------------------------
        # Node and edge feature embeddings
        # ----------------------------------------------------

        self.gene_emb = nn.Embedding(num_genes, 128)

        self.side_effect_emb = side_effect_emb
        self.side_effect_emb.weight.requires_grad = False

        self.side_effect_proj = nn.Linear(
            self.side_effect_emb.embedding_dim,
            128,
        )

        self.chem_proj = nn.Linear(chemical_feat_dim, 128)
        self.patient_proj = nn.Linear(patient_feat_dim, 128)
        self.demo_edge_proj = nn.Linear(demo_edge_feat_dim, 128)
        self.drug_proj = nn.Linear(768, 128)

        self.rept_emb = rept_emb
        self.mfr_emb = mfr_emb
        self.occr_emb = occr_emb
        self.sex_emb = sex_emb

        # ----------------------------------------------------
        # NNConv edge networks
        # ----------------------------------------------------

        chemical_edge_network = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128 * hidden_channels),
        )

        patient_drug_edge_network = nn.Sequential(
            nn.Linear(128, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 128 * hidden_channels),
        )

        # ----------------------------------------------------
        # Heterogeneous message passing
        # ----------------------------------------------------
        #
        # Deliberately excluded:
        #
        #   ('patient', 'targets', 'pt')
        #
        # The PT edges are labels and must not be used as input.
        # ----------------------------------------------------

        self.conv1 = HeteroConv(
            {
                (
                    "chemical",
                    "targets",
                    "chemical",
                ): NNConv(
                    in_channels=128,
                    out_channels=hidden_channels,
                    nn=chemical_edge_network,
                    aggr="mean",
                ),

                (
                    "chemical",
                    "rev_targets",
                    "chemical",
                ): NNConv(
                    in_channels=128,
                    out_channels=hidden_channels,
                    nn=chemical_edge_network,
                    aggr="mean",
                ),

                (
                    "chemical",
                    "targets",
                    "gene",
                ): SAGEConv(
                    (-1, -1),
                    hidden_channels,
                ),

                (
                    "gene",
                    "rev_targets",
                    "chemical",
                ): SAGEConv(
                    (-1, -1),
                    hidden_channels,
                ),

                (
                    "chemical",
                    "targets",
                    "disease_class",
                ): SAGEConv(
                    (-1, -1),
                    hidden_channels,
                ),

                (
                    "drugs",
                    "targets",
                    "chemical",
                ): SAGEConv(
                    (-1, -1),
                    hidden_channels,
                ),

                (
                    "chemical",
                    "rev_targets",
                    "drugs",
                ): SAGEConv(
                    (-1, -1),
                    hidden_channels,
                ),

                (
                    "drugs",
                    "targets",
                    "indi_pt",
                ): SAGEConv(
                    (-1, -1),
                    hidden_channels,
                ),

                (
                    "indi_pt",
                    "rev_targets",
                    "drugs",
                ): SAGEConv(
                    (-1, -1),
                    hidden_channels,
                ),

                (
                    "patient",
                    "targets",
                    "drugs",
                ): NNConv(
                    in_channels=128,
                    out_channels=hidden_channels,
                    nn=patient_drug_edge_network,
                    aggr="mean",
                ),

                (
                    "drugs",
                    "rev_targets",
                    "patient",
                ): SAGEConv(
                    (-1, -1),
                    hidden_channels,
                ),

                (
                    "gene",
                    "targets",
                    "gene",
                ): SAGEConv(
                    (-1, -1),
                    hidden_channels,
                ),

                (
                    "gene",
                    "rev_targets",
                    "gene",
                ): SAGEConv(
                    (-1, -1),
                    hidden_channels,
                ),
            },
            aggr="mean",
        )

        # ----------------------------------------------------
        # Patient-level classification head
        # ----------------------------------------------------
        #
        # Produces all label scores in one matrix multiplication:
        #
        # [batch_size, hidden_channels]
        #             @
        # [hidden_channels, num_selected_pts]
        #
        # -> [batch_size, num_selected_pts]
        # ----------------------------------------------------

        self.patient_norm = nn.LayerNorm(hidden_channels)
        self.patient_dropout = nn.Dropout(p=0.20)

        self.pt_classifier = nn.Linear(
            hidden_channels,
            num_selected_pts,
        )

    def forward(
        self,
        batch: HeteroData,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            seed_patient_logits:
                Shape [number_of_seed_patients, num_selected_pts].

            x_dict:
                Updated heterogeneous node embeddings.
        """

        device = next(self.parameters()).device

        # Shallow dictionary copies avoid overwriting batch.x_dict directly.
        x_dict = dict(batch.x_dict)
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = dict(batch.edge_attr_dict)

        # ----------------------------------------------------
        # Patient features
        # ----------------------------------------------------

        patient_raw = x_dict["patient"]

        categorical_indices = patient_raw[:, :4].long()
        numerical_features = patient_raw[:, 4:].float()

        rept_features = self.rept_emb(categorical_indices[:, 0])
        mfr_features = self.mfr_emb(categorical_indices[:, 1])
        occr_features = self.occr_emb(categorical_indices[:, 2])
        sex_features = self.sex_emb(categorical_indices[:, 3])

        patient_features = torch.cat(
            [
                rept_features,
                mfr_features,
                occr_features,
                sex_features,
                numerical_features,
            ],
            dim=1,
        )

        x_dict["patient"] = self.patient_proj(patient_features)

        # ----------------------------------------------------
        # Gene features
        # ----------------------------------------------------

        gene_indices = x_dict["gene"].long().view(-1)
        x_dict["gene"] = self.gene_emb(gene_indices)

        # ----------------------------------------------------
        # Chemical and drug features
        # ----------------------------------------------------

        x_dict["chemical"] = self.chem_proj(
            x_dict["chemical"].float()
        )

        x_dict["drugs"] = self.drug_proj(
            x_dict["drugs"].float()
        )

        # ----------------------------------------------------
        # Chemical-chemical edge embeddings
        # ----------------------------------------------------

        chemical_edge_type = (
            "chemical",
            "targets",
            "chemical",
        )

        reverse_chemical_edge_type = (
            "chemical",
            "rev_targets",
            "chemical",
        )

        if chemical_edge_type in edge_attr_dict:
            side_effect_ids = (
                edge_attr_dict[chemical_edge_type]
                .long()
                .view(-1)
                .to(device)
            )

            side_effect_features = self.side_effect_emb(
                side_effect_ids
            )

            edge_attr_dict[chemical_edge_type] = (
                self.side_effect_proj(side_effect_features)
            )

        if reverse_chemical_edge_type in edge_attr_dict:
            reverse_side_effect_ids = (
                edge_attr_dict[reverse_chemical_edge_type]
                .long()
                .view(-1)
                .to(device)
            )

            reverse_side_effect_features = self.side_effect_emb(
                reverse_side_effect_ids
            )

            edge_attr_dict[reverse_chemical_edge_type] = (
                self.side_effect_proj(reverse_side_effect_features)
            )

        # ----------------------------------------------------
        # Patient-drug edge features
        # ----------------------------------------------------

        patient_drug_edge_type = (
            "patient",
            "targets",
            "drugs",
        )

        if patient_drug_edge_type in edge_attr_dict:
            edge_attr_dict[patient_drug_edge_type] = (
                self.demo_edge_proj(
                    edge_attr_dict[patient_drug_edge_type].float()
                )
            )

        # ----------------------------------------------------
        # Message passing
        # ----------------------------------------------------

        updated_x_dict = self.conv1(
            x_dict,
            edge_index_dict,
            edge_attr_dict,
        )

        if "patient" not in updated_x_dict:
            raise RuntimeError(
                "The GNN did not produce updated patient embeddings. "
                "Check that the reverse drugs-to-patient edge exists."
            )

        patient_embeddings = updated_x_dict["patient"]

        # NeighborLoader places seed nodes first.
        seed_patient_count = int(batch["patient"].batch_size)

        seed_patient_embeddings = patient_embeddings[
            :seed_patient_count
        ]

        seed_patient_embeddings = self.patient_norm(
            seed_patient_embeddings
        )

        seed_patient_embeddings = F.relu(
            seed_patient_embeddings
        )

        seed_patient_embeddings = self.patient_dropout(
            seed_patient_embeddings
        )

        logits = self.pt_classifier(seed_patient_embeddings)

        return logits, updated_x_dict


# ============================================================
# Label extraction
# ============================================================

def extract_patient_pt_labels(
    data: HeteroData,
) -> Tuple[Dict[int, Set[int]], torch.Tensor, torch.Tensor]:
    """
    Extracts positive patient-to-PT edges from the original graph.

    Returns:
        patient_to_pt:
            Dictionary mapping global patient node IDs to global PT node IDs.

        patient_indices:
            Source patient IDs from the original patient-to-PT edge index.

        pt_indices:
            Destination PT IDs from the original patient-to-PT edge index.
    """

    edge_type = ("patient", "targets", "pt")

    if edge_type not in data.edge_types:
        raise KeyError(
            "The graph does not contain "
            "('patient', 'targets', 'pt') edges."
        )

    edge_index = data[edge_type].edge_index

    patient_indices = edge_index[0].long().cpu()
    pt_indices = edge_index[1].long().cpu()

    patient_to_pt: Dict[int, Set[int]] = defaultdict(set)

    for patient_id, pt_id in zip(
        patient_indices.tolist(),
        pt_indices.tolist(),
    ):
        patient_to_pt[patient_id].add(pt_id)

    return patient_to_pt, patient_indices, pt_indices


def split_patients(
    patient_ids: Sequence[int],
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Creates patient-disjoint training, validation, and test splits.
    """

    patient_ids_array = np.asarray(
        sorted(set(patient_ids)),
        dtype=np.int64,
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids_array)

    num_patients = len(patient_ids_array)

    train_end = int(num_patients * train_fraction)
    val_end = train_end + int(num_patients * val_fraction)

    train_patients = patient_ids_array[:train_end].tolist()
    val_patients = patient_ids_array[train_end:val_end].tolist()
    test_patients = patient_ids_array[val_end:].tolist()

    return train_patients, val_patients, test_patients


def select_top_pt_vocabulary(
    train_patient_ids: Sequence[int],
    patient_to_pt: Dict[int, Set[int]],
    num_selected_pts: int,
) -> Tuple[List[int], Dict[int, int], torch.Tensor]:
    """
    Selects the most frequent PT labels using training patients only.

    This avoids allowing validation or test label frequencies to influence
    vocabulary selection.

    Returns:
        selected_global_pt_ids:
            Global graph PT node IDs, ordered by descending training frequency.

        global_pt_to_class:
            Maps global PT node ID to classifier column.

        class_frequencies:
            Frequency of each selected class in the training set.
    """

    frequency: Dict[int, int] = defaultdict(int)

    for patient_id in train_patient_ids:
        for pt_id in patient_to_pt.get(patient_id, set()):
            frequency[pt_id] += 1

    sorted_pts = sorted(
        frequency.items(),
        key=lambda item: (-item[1], item[0]),
    )

    selected = sorted_pts[:num_selected_pts]

    selected_global_pt_ids = [
        pt_id for pt_id, _ in selected
    ]

    global_pt_to_class = {
        global_pt_id: class_index
        for class_index, global_pt_id
        in enumerate(selected_global_pt_ids)
    }

    class_frequencies = torch.tensor(
        [count for _, count in selected],
        dtype=torch.float,
    )

    return (
        selected_global_pt_ids,
        global_pt_to_class,
        class_frequencies,
    )


def convert_patient_labels_to_classes(
    patient_ids: Sequence[int],
    patient_to_pt: Dict[int, Set[int]],
    global_pt_to_class: Dict[int, int],
) -> Tuple[Dict[int, List[int]], List[int], int]:
    """
    Converts global PT IDs into classifier column indices.

    Patients with no labels in the selected vocabulary are excluded from the
    returned eligible-patient list.

    Returns:
        patient_to_classes:
            Global patient ID -> selected classifier columns.

        eligible_patients:
            Patients with at least one selected label.

        excluded_count:
            Number of patients whose labels all fell outside the vocabulary.
    """

    patient_to_classes: Dict[int, List[int]] = {}
    eligible_patients: List[int] = []

    excluded_count = 0

    for patient_id in patient_ids:
        classes = sorted(
            global_pt_to_class[pt_id]
            for pt_id in patient_to_pt.get(patient_id, set())
            if pt_id in global_pt_to_class
        )

        patient_to_classes[patient_id] = classes

        if classes:
            eligible_patients.append(patient_id)
        else:
            excluded_count += 1

    return patient_to_classes, eligible_patients, excluded_count


# ============================================================
# Graph preparation
# ============================================================

def remove_target_edges(
    data: HeteroData,
) -> HeteroData:
    """
    Removes every edge type directly connecting patient and PT nodes.

    This is essential. Keeping patient-to-PT labels in the message-passing
    graph would allow the model to observe the answer it is being trained to
    predict.
    """

    graph = copy.copy(data)

    edge_types_to_remove = []

    for edge_type in graph.edge_types:
        source_type, _, destination_type = edge_type

        is_patient_pt_edge = {
            source_type,
            destination_type,
        } == {"patient", "pt"}

        if is_patient_pt_edge:
            edge_types_to_remove.append(edge_type)

    for edge_type in edge_types_to_remove:
        del graph[edge_type]

    print(
        "Removed target edge types:",
        edge_types_to_remove,
    )

    return graph


def create_neighbor_loader(
    graph: HeteroData,
    patient_ids: Sequence[int],
    batch_size: int,
    num_neighbors: Sequence[int],
    shuffle: bool,
) -> NeighborLoader:
    """
    Creates patient-centred batches.

    Each batch is seeded by patient nodes rather than patient-PT edges.
    """

    patient_tensor = torch.tensor(
        patient_ids,
        dtype=torch.long,
    )

    neighbor_configuration = {
        edge_type: list(num_neighbors)
        for edge_type in graph.edge_types
    }

    return NeighborLoader(
        graph,
        input_nodes=("patient", patient_tensor),
        num_neighbors=neighbor_configuration,
        batch_size=batch_size,
        shuffle=shuffle,
    )


# ============================================================
# Target construction
# ============================================================

def build_dense_targets(
    global_patient_ids: torch.Tensor,
    patient_to_classes: Dict[int, List[int]],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Constructs a dense multilabel target matrix.

    Shape:
        [batch_size, num_classes]

    With 1,000 classes and a batch of 400, this is only 400,000 values.
    """

    targets = torch.zeros(
        (global_patient_ids.numel(), num_classes),
        dtype=torch.float32,
        device=device,
    )

    for row, patient_id in enumerate(
        global_patient_ids.tolist()
    ):
        class_indices = patient_to_classes.get(
            int(patient_id),
            [],
        )

        if class_indices:
            targets[
                row,
                torch.tensor(
                    class_indices,
                    dtype=torch.long,
                    device=device,
                ),
            ] = 1.0

    return targets


# ============================================================
# Hard-negative loss
# ============================================================

def positive_and_hard_negative_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_hard_negatives: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes BCE using:

        - every positive label;
        - the top-N highest-scoring incorrect labels for each patient.

    Hard-negative selection is based on detached logits so that selecting the
    indices does not become part of the gradient graph.

    The positive and negative losses are averaged separately and then combined.
    This prevents the 50 negatives from automatically outweighing a patient's
    small number of positives.
    """

    positive_mask = targets.bool()
    negative_mask = ~positive_mask

    # Positive BCE.
    positive_logits = logits[positive_mask]

    if positive_logits.numel() == 0:
        raise RuntimeError(
            "The current batch contains no positive labels."
        )

    positive_targets = torch.ones_like(positive_logits)

    positive_loss = F.binary_cross_entropy_with_logits(
        positive_logits,
        positive_targets,
    )

    # Set all positives to -infinity before hard-negative selection.
    negative_selection_scores = logits.detach().masked_fill(
        positive_mask,
        float("-inf"),
    )

    available_negatives = negative_mask.sum(dim=1)
    maximum_available = int(
        available_negatives.min().item()
    )

    hard_negative_count = min(
        num_hard_negatives,
        maximum_available,
        logits.size(1),
    )

    if hard_negative_count <= 0:
        return positive_loss, {
            "positive_loss": float(positive_loss.detach()),
            "negative_loss": 0.0,
            "hard_negatives_per_patient": 0.0,
        }

    hard_negative_indices = torch.topk(
        negative_selection_scores,
        k=hard_negative_count,
        dim=1,
    ).indices

    hard_negative_logits = torch.gather(
        logits,
        dim=1,
        index=hard_negative_indices,
    )

    hard_negative_targets = torch.zeros_like(
        hard_negative_logits
    )

    hard_negative_loss = F.binary_cross_entropy_with_logits(
        hard_negative_logits,
        hard_negative_targets,
    )

    loss = positive_loss + hard_negative_loss

    return loss, {
        "positive_loss": float(positive_loss.detach()),
        "negative_loss": float(hard_negative_loss.detach()),
        "hard_negatives_per_patient": float(
            hard_negative_count
        ),
    }


# ============================================================
# Ranking metrics
# ============================================================

def batch_recall_at_k(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> Tuple[float, int]:
    """
    Returns the sum of per-patient Recall@K and the number of valid patients.
    """

    k = min(k, logits.size(1))

    topk_indices = torch.topk(
        logits,
        k=k,
        dim=1,
    ).indices

    topk_hits = torch.gather(
        targets,
        dim=1,
        index=topk_indices,
    ).sum(dim=1)

    positive_counts = targets.sum(dim=1)
    valid_mask = positive_counts > 0

    recalls = topk_hits[valid_mask] / positive_counts[valid_mask]

    return float(recalls.sum().item()), int(valid_mask.sum().item())


def batch_ndcg_at_k(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> Tuple[float, int]:
    """
    Calculates binary-relevance NDCG@K.

    For each patient:

        DCG = sum(relevance_i / log2(rank_i + 1))

    The ideal ranking places all true labels before all false labels.
    """

    k = min(k, logits.size(1))

    topk_indices = torch.topk(
        logits,
        k=k,
        dim=1,
    ).indices

    ranked_relevance = torch.gather(
        targets,
        dim=1,
        index=topk_indices,
    )

    discounts = 1.0 / torch.log2(
        torch.arange(
            2,
            k + 2,
            device=logits.device,
            dtype=torch.float32,
        )
    )

    dcg = (
        ranked_relevance * discounts.unsqueeze(0)
    ).sum(dim=1)

    positive_counts = targets.sum(dim=1).long()
    valid_mask = positive_counts > 0

    capped_positive_counts = torch.clamp(
        positive_counts,
        max=k,
    )

    cumulative_discounts = torch.cumsum(
        discounts,
        dim=0,
    )

    ideal_dcg = cumulative_discounts[
        capped_positive_counts.clamp(min=1) - 1
    ]

    ndcg = dcg[valid_mask] / ideal_dcg[valid_mask]

    return float(ndcg.sum().item()), int(valid_mask.sum().item())


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model: AdvGNNPatientClassifier,
    loader: NeighborLoader,
    optimiser: torch.optim.Optimizer,
    patient_to_classes: Dict[int, List[int]],
    num_classes: int,
    num_hard_negatives: int,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_positive_loss = 0.0
    total_negative_loss = 0.0
    total_batches = 0

    progress = tqdm(
        loader,
        desc=f"Training epoch {epoch}",
    )

    for batch in progress:
        batch = batch.to(device)

        optimiser.zero_grad(set_to_none=True)

        logits, _ = model(batch)

        seed_count = int(batch["patient"].batch_size)

        global_patient_ids = batch["patient"].n_id[
            :seed_count
        ].cpu()

        targets = build_dense_targets(
            global_patient_ids=global_patient_ids,
            patient_to_classes=patient_to_classes,
            num_classes=num_classes,
            device=device,
        )

        loss, loss_components = positive_and_hard_negative_bce(
            logits=logits,
            targets=targets,
            num_hard_negatives=num_hard_negatives,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimiser.step()

        total_loss += float(loss.detach())
        total_positive_loss += loss_components["positive_loss"]
        total_negative_loss += loss_components["negative_loss"]
        total_batches += 1

        progress.set_postfix(
            loss=f"{float(loss.detach()):.4f}",
            positive=f"{loss_components['positive_loss']:.4f}",
            negative=f"{loss_components['negative_loss']:.4f}",
        )

        del batch, logits, targets, loss

    if total_batches == 0:
        raise RuntimeError("The training loader produced no batches.")

    return {
        "loss": total_loss / total_batches,
        "positive_loss": total_positive_loss / total_batches,
        "negative_loss": total_negative_loss / total_batches,
    }


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_model(
    model: AdvGNNPatientClassifier,
    loader: NeighborLoader,
    patient_to_classes: Dict[int, List[int]],
    num_classes: int,
    device: torch.device,
    description: str,
) -> Dict[str, float]:
    """
    Evaluates every patient against the entire selected PT vocabulary.
    """

    model.eval()

    recall_sums = {
        k: 0.0 for k in RECALL_K_VALUES
    }

    recall_counts = {
        k: 0 for k in RECALL_K_VALUES
    }

    ndcg_sums = {
        k: 0.0 for k in RECALL_K_VALUES
    }

    ndcg_counts = {
        k: 0 for k in RECALL_K_VALUES
    }

    for batch in tqdm(loader, desc=description):
        batch = batch.to(device)

        logits, _ = model(batch)

        seed_count = int(batch["patient"].batch_size)

        global_patient_ids = batch["patient"].n_id[
            :seed_count
        ].cpu()

        targets = build_dense_targets(
            global_patient_ids=global_patient_ids,
            patient_to_classes=patient_to_classes,
            num_classes=num_classes,
            device=device,
        )

        for k in RECALL_K_VALUES:
            recall_sum, recall_count = batch_recall_at_k(
                logits,
                targets,
                k,
            )

            ndcg_sum, ndcg_count = batch_ndcg_at_k(
                logits,
                targets,
                k,
            )

            recall_sums[k] += recall_sum
            recall_counts[k] += recall_count

            ndcg_sums[k] += ndcg_sum
            ndcg_counts[k] += ndcg_count

        del batch, logits, targets

    metrics: Dict[str, float] = {}

    for k in RECALL_K_VALUES:
        metrics[f"Recall@{k}"] = (
            recall_sums[k] / recall_counts[k]
            if recall_counts[k] > 0
            else 0.0
        )

        metrics[f"NDCG@{k}"] = (
            ndcg_sums[k] / ndcg_counts[k]
            if ndcg_counts[k] > 0
            else 0.0
        )

    metrics["evaluated_patients"] = float(
        recall_counts[RECALL_K_VALUES[0]]
    )

    return metrics


@torch.no_grad()
def evaluate_frequency_baseline(
    loader: NeighborLoader,
    patient_to_classes: Dict[int, List[int]],
    class_frequencies: torch.Tensor,
    num_classes: int,
    device: torch.device,
    description: str,
) -> Dict[str, float]:
    """
    Baseline that assigns every patient the same ranking:

        most frequent training PT first,
        second-most frequent training PT second,
        etc.
    """

    frequency_logits = class_frequencies.to(device).float()

    recall_sums = {
        k: 0.0 for k in RECALL_K_VALUES
    }

    recall_counts = {
        k: 0 for k in RECALL_K_VALUES
    }

    ndcg_sums = {
        k: 0.0 for k in RECALL_K_VALUES
    }

    ndcg_counts = {
        k: 0 for k in RECALL_K_VALUES
    }

    for batch in tqdm(loader, desc=description):
        seed_count = int(batch["patient"].batch_size)

        global_patient_ids = batch["patient"].n_id[
            :seed_count
        ].cpu()

        targets = build_dense_targets(
            global_patient_ids=global_patient_ids,
            patient_to_classes=patient_to_classes,
            num_classes=num_classes,
            device=device,
        )

        baseline_logits = frequency_logits.unsqueeze(0).expand(
            seed_count,
            -1,
        )

        for k in RECALL_K_VALUES:
            recall_sum, recall_count = batch_recall_at_k(
                baseline_logits,
                targets,
                k,
            )

            ndcg_sum, ndcg_count = batch_ndcg_at_k(
                baseline_logits,
                targets,
                k,
            )

            recall_sums[k] += recall_sum
            recall_counts[k] += recall_count

            ndcg_sums[k] += ndcg_sum
            ndcg_counts[k] += ndcg_count

    metrics: Dict[str, float] = {}

    for k in RECALL_K_VALUES:
        metrics[f"Recall@{k}"] = (
            recall_sums[k] / recall_counts[k]
            if recall_counts[k] > 0
            else 0.0
        )

        metrics[f"NDCG@{k}"] = (
            ndcg_sums[k] / ndcg_counts[k]
            if ndcg_counts[k] > 0
            else 0.0
        )

    metrics["evaluated_patients"] = float(
        recall_counts[RECALL_K_VALUES[0]]
    )

    return metrics


# ============================================================
# Checkpoint helpers
# ============================================================

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    epoch: int,
    selected_global_pt_ids: Sequence[int],
    class_frequencies: torch.Tensor,
    validation_metrics: Dict[str, float],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "selected_global_pt_ids": list(selected_global_pt_ids),
        "class_frequencies": class_frequencies.cpu(),
        "validation_metrics": validation_metrics,
        "num_selected_pts": len(selected_global_pt_ids),
        "num_hard_negatives": NUM_HARD_NEGATIVES,
    }

    torch.save(checkpoint, path)


def load_encoder_checkpoint_if_requested(
    model: nn.Module,
    checkpoint_path: Optional[Path],
    device: torch.device,
) -> None:
    """
    Optionally loads compatible parameters from an older model.

    The old edge-prediction checkpoint is not directly compatible with the new
    classifier head. strict=False permits compatible encoder parameters to load
    while leaving the new classification head randomly initialised.
    """

    if checkpoint_path is None:
        return

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    loaded_object = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_state_dict" in loaded_object:
        state_dict = loaded_object["model_state_dict"]
    else:
        state_dict = loaded_object

    incompatible = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print("Loaded compatible checkpoint parameters.")
    print("Missing keys:", incompatible.missing_keys)
    print("Unexpected keys:", incompatible.unexpected_keys)


# ============================================================
# Reporting helpers
# ============================================================

def print_metrics(
    heading: str,
    metrics: Dict[str, float],
) -> None:
    print(f"\n{heading}")

    for name, value in metrics.items():
        if name == "evaluated_patients":
            print(f"  {name}: {int(value)}")
        else:
            print(f"  {name}: {value:.6f}")


def calculate_label_coverage(
    patient_ids: Sequence[int],
    patient_to_pt: Dict[int, Set[int]],
    selected_pt_ids: Set[int],
) -> Dict[str, float]:
    """
    Measures how much of the original label data remains after restricting the
    model to the selected vocabulary.
    """

    total_labels = 0
    covered_labels = 0
    patients_with_covered_label = 0

    for patient_id in patient_ids:
        original_labels = patient_to_pt.get(
            patient_id,
            set(),
        )

        covered = original_labels.intersection(
            selected_pt_ids
        )

        total_labels += len(original_labels)
        covered_labels += len(covered)

        if covered:
            patients_with_covered_label += 1

    return {
        "label_occurrence_coverage": (
            covered_labels / total_labels
            if total_labels > 0
            else 0.0
        ),
        "patient_coverage": (
            patients_with_covered_label / len(patient_ids)
            if patient_ids
            else 0.0
        ),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    set_seed(SEED)

    cwd = Path.cwd()

    graph_path = (
        cwd
        / "output_csv"
        / "graph_data"
        / "graph_data.pt"
    )

    demo_path = (
        cwd
        / "output_csv"
        / "demo_clean"
        / "demo_clean.csv"
    )

    embedding_path = (
        cwd
        / "output_csv"
        / "pre-embeddings"
        / "pre_embeddings.pt"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data: HeteroData = torch.load(
        graph_path,
        map_location="cpu",
    )

    demo = pd.read_csv(demo_path)

    embedding_map = torch.load(
        embedding_path,
        map_location="cpu",
    )

    print("Total graph PT nodes:", data["pt"].num_nodes)
    print("Total graph patients:", data["patient"].num_nodes)

    # --------------------------------------------------------
    # Extract labels before removing target edges
    # --------------------------------------------------------

    (
        patient_to_pt,
        _,
        _,
    ) = extract_patient_pt_labels(data)

    labelled_patient_ids = sorted(
        patient_to_pt.keys()
    )

    print(
        "Patients with at least one original PT:",
        len(labelled_patient_ids),
    )

    # --------------------------------------------------------
    # Patient-disjoint split
    # --------------------------------------------------------

    (
        train_patient_ids,
        val_patient_ids,
        test_patient_ids,
    ) = split_patients(
        patient_ids=labelled_patient_ids,
        train_fraction=TRAIN_FRACTION,
        val_fraction=VAL_FRACTION,
        seed=SEED,
    )

    print("\nPatient split")
    print("  Training patients:", len(train_patient_ids))
    print("  Validation patients:", len(val_patient_ids))
    print("  Test patients:", len(test_patient_ids))

    # --------------------------------------------------------
    # Select top labels from training patients only
    # --------------------------------------------------------

    (
        selected_global_pt_ids,
        global_pt_to_class,
        class_frequencies,
    ) = select_top_pt_vocabulary(
        train_patient_ids=train_patient_ids,
        patient_to_pt=patient_to_pt,
        num_selected_pts=NUM_SELECTED_PTS,
    )

    actual_num_classes = len(selected_global_pt_ids)

    if actual_num_classes == 0:
        raise RuntimeError(
            "No PT labels were found among the training patients."
        )

    print(
        "\nSelected PT vocabulary size:",
        actual_num_classes,
    )

    selected_pt_set = set(selected_global_pt_ids)

    # --------------------------------------------------------
    # Convert labels into classifier indices
    # --------------------------------------------------------

    (
        train_patient_to_classes,
        eligible_train_patients,
        excluded_train_patients,
    ) = convert_patient_labels_to_classes(
        patient_ids=train_patient_ids,
        patient_to_pt=patient_to_pt,
        global_pt_to_class=global_pt_to_class,
    )

    (
        val_patient_to_classes,
        eligible_val_patients,
        excluded_val_patients,
    ) = convert_patient_labels_to_classes(
        patient_ids=val_patient_ids,
        patient_to_pt=patient_to_pt,
        global_pt_to_class=global_pt_to_class,
    )

    (
        test_patient_to_classes,
        eligible_test_patients,
        excluded_test_patients,
    ) = convert_patient_labels_to_classes(
        patient_ids=test_patient_ids,
        patient_to_pt=patient_to_pt,
        global_pt_to_class=global_pt_to_class,
    )

    print("\nPatients retained after vocabulary restriction")
    print(
        f"  Train: {len(eligible_train_patients)} "
        f"retained, {excluded_train_patients} excluded"
    )
    print(
        f"  Validation: {len(eligible_val_patients)} "
        f"retained, {excluded_val_patients} excluded"
    )
    print(
        f"  Test: {len(eligible_test_patients)} "
        f"retained, {excluded_test_patients} excluded"
    )

    for split_name, split_patient_ids in [
        ("Train", train_patient_ids),
        ("Validation", val_patient_ids),
        ("Test", test_patient_ids),
    ]:
        coverage = calculate_label_coverage(
            patient_ids=split_patient_ids,
            patient_to_pt=patient_to_pt,
            selected_pt_ids=selected_pt_set,
        )

        print(
            f"  {split_name} label-occurrence coverage: "
            f"{coverage['label_occurrence_coverage']:.4f}"
        )

        print(
            f"  {split_name} patient coverage: "
            f"{coverage['patient_coverage']:.4f}"
        )

    # --------------------------------------------------------
    # Prepare patient features
    # --------------------------------------------------------

    demo["rept_cod"] = demo["rept_cod"].fillna(
        "UNKNOWN"
    )

    demo["mfr_sndr"] = demo["mfr_sndr"].fillna(
        "UNKNOWN"
    )

    demo["occr_country"] = demo["occr_country"].fillna(
        "UNKNOWN"
    )

    valid_sex_values = {"F", "M", "UNK"}

    demo["sex"] = demo["sex"].fillna("UNK")
    demo["sex"] = demo["sex"].apply(
        lambda value: (
            value if value in valid_sex_values else "UNK"
        )
    )

    rept_emb = nn.Embedding(
        len(demo["rept_cod"].unique()),
        4,
    )

    mfr_emb = nn.Embedding(
        len(demo["mfr_sndr"].unique()),
        8,
    )

    occr_emb = nn.Embedding(
        len(demo["occr_country"].unique()),
        6,
    )

    sex_emb = nn.Embedding(
        3,
        4,
    )

    patient_categorical = torch.as_tensor(
        data["patient"].cat_index,
        dtype=torch.long,
    )

    patient_numerical = torch.as_tensor(
        data["patient"].numerical,
        dtype=torch.float32,
    )

    data["patient"].x = torch.cat(
        [
            patient_categorical,
            patient_numerical,
        ],
        dim=1,
    )

    patient_projected_feature_dim = (
        rept_emb.embedding_dim
        + mfr_emb.embedding_dim
        + occr_emb.embedding_dim
        + sex_emb.embedding_dim
        + patient_numerical.size(1)
    )

    # --------------------------------------------------------
    # Side-effect embeddings
    # --------------------------------------------------------

    side_effect_tensor = torch.stack(
        list(embedding_map["side_effect"].values())
    ).float()

    side_effect_emb = nn.Embedding.from_pretrained(
        side_effect_tensor,
        freeze=True,
    )

    # --------------------------------------------------------
    # Remove patient-PT target edges
    # --------------------------------------------------------

    message_passing_graph = remove_target_edges(data)

    if (
        "patient",
        "targets",
        "pt",
    ) in message_passing_graph.edge_types:
        raise RuntimeError(
            "Patient-to-PT target edges were not removed."
        )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = create_neighbor_loader(
        graph=message_passing_graph,
        patient_ids=eligible_train_patients,
        batch_size=TRAIN_BATCH_SIZE,
        num_neighbors=TRAIN_NUM_NEIGHBORS,
        shuffle=True,
    )

    val_loader = create_neighbor_loader(
        graph=message_passing_graph,
        patient_ids=eligible_val_patients,
        batch_size=EVAL_BATCH_SIZE,
        num_neighbors=EVAL_NUM_NEIGHBORS,
        shuffle=False,
    )

    test_loader = create_neighbor_loader(
        graph=message_passing_graph,
        patient_ids=eligible_test_patients,
        batch_size=EVAL_BATCH_SIZE,
        num_neighbors=EVAL_NUM_NEIGHBORS,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    model = AdvGNNPatientClassifier(
        hidden_channels=HIDDEN_CHANNELS,
        patient_feat_dim=patient_projected_feature_dim,
        chemical_feat_dim=data["chemical"].x.size(1),
        demo_edge_feat_dim=data[
            "patient",
            "targets",
            "drugs",
        ].edge_attr.size(1),
        num_genes=data["gene"].num_nodes,
        num_selected_pts=actual_num_classes,
        side_effect_emb=side_effect_emb,
        rept_emb=rept_emb,
        mfr_emb=mfr_emb,
        occr_emb=occr_emb,
        sex_emb=sex_emb,
    ).to(device)

    load_encoder_checkpoint_if_requested(
        model=model,
        checkpoint_path=CHECKPOINT_TO_LOAD,
        device=device,
    )

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Frequency baseline
    # --------------------------------------------------------

    val_baseline_metrics = evaluate_frequency_baseline(
        loader=val_loader,
        patient_to_classes=val_patient_to_classes,
        class_frequencies=class_frequencies,
        num_classes=actual_num_classes,
        device=device,
        description="Validation frequency baseline",
    )

    test_baseline_metrics = evaluate_frequency_baseline(
        loader=test_loader,
        patient_to_classes=test_patient_to_classes,
        class_frequencies=class_frequencies,
        num_classes=actual_num_classes,
        device=device,
        description="Test frequency baseline",
    )

    print_metrics(
        "Validation most-common-PT baseline",
        val_baseline_metrics,
    )

    print_metrics(
        "Test most-common-PT baseline",
        test_baseline_metrics,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_validation_ndcg = float("-inf")
    best_epoch = -1

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimiser=optimiser,
            patient_to_classes=train_patient_to_classes,
            num_classes=actual_num_classes,
            num_hard_negatives=NUM_HARD_NEGATIVES,
            device=device,
            epoch=epoch,
        )

        validation_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            patient_to_classes=val_patient_to_classes,
            num_classes=actual_num_classes,
            device=device,
            description=f"Validation epoch {epoch}",
        )

        print_metrics(
            f"Epoch {epoch} training losses",
            train_metrics,
        )

        print_metrics(
            f"Epoch {epoch} validation metrics",
            validation_metrics,
        )

        validation_ndcg = validation_metrics["NDCG@50"]

        if validation_ndcg > best_validation_ndcg:
            best_validation_ndcg = validation_ndcg
            best_epoch = epoch

            save_checkpoint(
                path=OUTPUT_CHECKPOINT,
                model=model,
                optimiser=optimiser,
                epoch=epoch,
                selected_global_pt_ids=selected_global_pt_ids,
                class_frequencies=class_frequencies,
                validation_metrics=validation_metrics,
            )

            print(
                f"Saved new best checkpoint at epoch {epoch}: "
                f"{OUTPUT_CHECKPOINT}"
            )

    print(
        f"\nBest epoch: {best_epoch}; "
        f"best validation NDCG@50: "
        f"{best_validation_ndcg:.6f}"
    )

    # --------------------------------------------------------
    # Load best model and evaluate on test patients
    # --------------------------------------------------------

    best_checkpoint = torch.load(
        OUTPUT_CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    test_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        patient_to_classes=test_patient_to_classes,
        num_classes=actual_num_classes,
        device=device,
        description="Final test evaluation",
    )

    print_metrics(
        "Final model test metrics",
        test_metrics,
    )

    print_metrics(
        "Most-common-PT test baseline",
        test_baseline_metrics,
    )

    print("\nModel improvement over frequency baseline")

    for metric_name in [
        "Recall@10",
        "Recall@50",
        "NDCG@10",
        "NDCG@50",
    ]:
        improvement = (
            test_metrics[metric_name]
            - test_baseline_metrics[metric_name]
        )

        print(
            f"  {metric_name}: {improvement:+.6f}"
        )


if __name__ == "__main__":
    main()