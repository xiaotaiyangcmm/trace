"""Self-contained LongNeuronSet loader for TraceFormer/Neuron-DPO.

Both training and evaluation use the same feature construction and candidate
index convention.  All paths are supplied by the caller; this module contains
no references to the original project or datasets.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_pickle(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _fps(points: np.ndarray, npoint: int, seed: int) -> np.ndarray:
    """Deterministic farthest-point sampling used by both train and test."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        raise ValueError(f"Invalid point cloud shape: {points.shape}")
    if len(points) < npoint:
        repeats = int(np.ceil(npoint / len(points)))
        points = np.tile(points, (repeats, 1))

    xyz = points[:, :3]
    selected = np.empty(npoint, dtype=np.int64)
    distances = np.full(len(points), np.inf, dtype=np.float64)
    farthest = int(seed % len(points))
    for index in range(npoint):
        selected[index] = farthest
        centroid = xyz[farthest]
        current = np.sum((xyz - centroid) ** 2, axis=1)
        distances = np.minimum(distances, current)
        farthest = int(np.argmax(distances))
    return points[selected]


class TracingDataset(Dataset):
    """Expands neuron sequences into individual tracing states.

    Each state contains the complete available ordered tracing history, the
    ground-truth continuation, and every generated candidate.  Candidate index
    zero is the annotated next segment before candidate permutation.
    """

    def __init__(
        self,
        sequence_path: str | Path,
        point_path: str | Path,
        npoint: int = 256,
    ) -> None:
        self.sequence_path = Path(sequence_path)
        self.point_path = Path(point_path)
        self.npoint = int(npoint)
        self.records = _load_pickle(self.sequence_path)
        self.point_dict: Dict[Any, np.ndarray] = _load_pickle(self.point_path)
        self.states: List[tuple[int, int]] = []

        # Start at two observed segments, matching the Neuron-DPO training
        # setup used by the original full-history loader.
        for record_index, record in enumerate(self.records):
            segment_count = len(record[0])
            candidate_lists = record[2]
            history_image_embeddings = record[4]
            for history_length in range(2, segment_count):
                candidate_index = history_length - 1
                if (
                    candidate_index < len(candidate_lists)
                    and candidate_index < len(history_image_embeddings)
                    and len(candidate_lists[candidate_index]) > 0
                    and len(history_image_embeddings[candidate_index]) > 0
                ):
                    self.states.append((record_index, history_length))

        if not self.states:
            raise RuntimeError(f"No valid tracing states found in {self.sequence_path}")

    def __len__(self) -> int:
        return len(self.states)

    def _point_cloud(self, segment_id: Any) -> np.ndarray:
        candidates = (segment_id, str(segment_id), int(segment_id))
        for key in candidates:
            if key in self.point_dict:
                return np.asarray(self.point_dict[key], dtype=np.float32)
        raise KeyError(f"Point cloud not found for segment {segment_id}")

    def _edge_points(self, source: Any, target: Any) -> np.ndarray:
        combined = np.concatenate(
            [self._point_cloud(source), self._point_cloud(target)], axis=0
        )
        # Preserve the stochastic FPS augmentation used by the released code.
        seed = int(np.random.randint(0, max(1, len(combined))))
        return _fps(combined, self.npoint, seed)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record_index, history_length = self.states[index]
        record = self.records[record_index]

        full_sequence: Sequence[Any] = record[0]
        history_coordinates = np.asarray(record[1][:history_length], dtype=np.float32)
        negative_candidate_ids = list(record[2][history_length - 1])
        candidate_image_raw = record[3][history_length - 1]
        positive_image_raw = record[4][history_length - 1]

        observed_with_target = list(full_sequence[: history_length + 1])
        target_id = observed_with_target[-1]
        observed_ids = observed_with_target[:-1]
        center_id = observed_ids[-1]
        candidate_ids = [target_id] + negative_candidate_ids

        history_points = [
            self._edge_points(observed_with_target[i - 1], observed_with_target[i])
            for i in range(1, len(observed_with_target) - 1)
        ]
        candidate_points = [self._edge_points(center_id, candidate_id) for candidate_id in candidate_ids]

        if not history_points:
            raise RuntimeError("A tracing state must contain at least one history edge")

        center = np.mean(history_coordinates, axis=0)
        all_points = np.concatenate(history_points + candidate_points, axis=0)
        scale = float(np.max(np.linalg.norm(all_points[:, :3] - center, axis=1)))
        scale = max(scale, 1e-12)
        history_points = (np.stack(history_points) - center) / scale
        candidate_points = (np.stack(candidate_points) - center) / scale
        normalized_coordinates = (history_coordinates - center) / scale

        candidate_images = np.concatenate(
            [np.asarray(item).reshape(1, -1) for item in candidate_image_raw], axis=0
        )
        positive_images = np.concatenate(
            [np.asarray(item).reshape(1, -1) for item in positive_image_raw], axis=0
        )
        combined_images = np.concatenate([positive_images, candidate_images], axis=0)
        candidate_count = len(candidate_ids)
        history_node_images = combined_images[:-candidate_count]
        candidate_node_images = combined_images[-candidate_count:]

        history_edge_images = np.stack(
            [
                np.concatenate([history_node_images[i - 1], history_node_images[i]])
                for i in range(1, len(history_node_images))
            ]
        )
        candidate_edge_images = np.stack(
            [
                np.concatenate([history_node_images[-1], candidate_node_images[i]])
                for i in range(candidate_count)
            ]
        )

        # Individual 128-D node views used by the original TraceFormer image
        # projection when it is evaluated on the Neuron-DPO state list.
        traceformer_history_images = history_node_images[:-1]
        traceformer_candidate_images = candidate_node_images

        permutation = np.random.permutation(candidate_count)
        candidate_points = candidate_points[permutation]
        candidate_edge_images = candidate_edge_images[permutation]
        traceformer_candidate_images = traceformer_candidate_images[permutation]
        candidate_ids = [candidate_ids[position] for position in permutation]
        gt_index = int(np.where(permutation == 0)[0][0])

        # ``labels`` are only used by the legacy model to identify the boundary
        # between history and candidates.  Its required length is one greater
        # than the number of history-edge tokens.
        labels = np.asarray(observed_ids, dtype=np.int64)

        return {
            "history_ids": [int(value) for value in observed_ids],
            "candidate_ids": [int(value) for value in candidate_ids],
            "history_points": torch.from_numpy(history_points.astype(np.float32)),
            "candidate_points": torch.from_numpy(candidate_points.astype(np.float32)),
            "history_images": torch.from_numpy(history_edge_images.astype(np.float32)),
            "candidate_images": torch.from_numpy(candidate_edge_images.astype(np.float32)),
            "traceformer_history_images": torch.from_numpy(
                traceformer_history_images.astype(np.float32)
            ),
            "traceformer_candidate_images": torch.from_numpy(
                traceformer_candidate_images.astype(np.float32)
            ),
            "coordinates": torch.from_numpy(normalized_coordinates.astype(np.float32)),
            "labels": torch.from_numpy(labels),
            "history_length": int(history_length),
            "sequence_id": int(record_index),
            "gt_index": gt_index,
        }


def single_state_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("Variable-sized tracing states require batch_size=1")
    return batch[0]


class LegacyEvaluationDataset(TracingDataset):
    """Exact evaluation preprocessing used by the released paper checkpoint.

    The released inference code evaluates the latest two traced connections and
    includes the ground-truth continuation edge at prediction index one.  This
    class preserves that convention while accepting only package-local paths.
    """

    def __init__(self, sequence_path: str | Path, point_path: str | Path, npoint: int = 256):
        super().__init__(sequence_path, point_path, npoint)
        self.states = []
        for record_index, record in enumerate(self.records):
            segment_count = len(record[0])
            for history_length in range(1, segment_count):
                candidate_index = history_length - 1
                if (
                    candidate_index < len(record[2])
                    and candidate_index < len(record[4])
                    and len(record[2][candidate_index]) > 0
                    and len(record[4][candidate_index]) > 0
                ):
                    self.states.append((record_index, history_length))

    def _legacy_edge_points(self, source: Any, target: Any) -> np.ndarray:
        points = np.concatenate([self._point_cloud(source), self._point_cloud(target)], axis=0)
        xyz = points[:, :3]
        selected = np.empty(self.npoint, dtype=np.int64)
        distances = np.full(len(points), 1e10, dtype=np.float64)
        farthest = int(np.random.randint(0, len(points)))
        for sample_index in range(self.npoint):
            selected[sample_index] = farthest
            centroid = xyz[farthest]
            current = np.sum((xyz - centroid) ** 2, axis=1)
            distances[current < distances] = current[current < distances]
            farthest = int(np.argmax(distances))
        return points[selected]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record_index, history_length = self.states[index]
        record = self.records[record_index]

        full_sequence = record[0]
        observed_with_target = list(full_sequence[: history_length + 1][-3:])
        if len(observed_with_target) < 2:
            raise RuntimeError("Invalid evaluation state with fewer than two segments")

        coordinates = np.asarray(record[1][:history_length][-2:], dtype=np.float32)
        center = np.mean(coordinates, axis=0)
        target_id = observed_with_target[-1]
        candidate_ids = [target_id] + list(record[2][history_length - 1])

        history_points = []
        candidate_points = []
        all_points = []
        for edge_index in range(1, len(observed_with_target)):
            edge = self._legacy_edge_points(
                observed_with_target[edge_index - 1], observed_with_target[edge_index]
            )
            if edge_index == len(observed_with_target) - 1:
                candidate_points.append(edge)
                all_points.append(edge)
            history_points.append(edge)
            all_points.append(edge)

        source_id = observed_with_target[-2]
        for candidate_id in candidate_ids[1:]:
            edge = self._legacy_edge_points(source_id, candidate_id)
            candidate_points.append(edge)
            all_points.append(edge)

        scale = float(np.max(np.linalg.norm(np.concatenate(all_points)[:, :3] - center, axis=1)))
        scale = max(scale, 1e-12)
        history_points = (np.stack(history_points) - center) / scale
        candidate_points = (np.stack(candidate_points) - center) / scale
        coordinates = (coordinates - center) / scale

        candidate_image_raw = record[3][history_length - 1]
        positive_image_raw = record[4][history_length - 1][-3:]
        candidate_images = np.concatenate(
            [np.asarray(item).reshape(1, -1) for item in candidate_image_raw], axis=0
        )
        positive_images = np.concatenate(
            [np.asarray(item).reshape(1, -1) for item in positive_image_raw], axis=0
        )
        combined_images = np.concatenate([positive_images, candidate_images], axis=0)
        candidate_count = len(candidate_ids)
        history_images = combined_images[:-candidate_count]
        candidate_images = combined_images[-candidate_count:]

        observed_ids = observed_with_target[:-1]
        # The released model slices predictions from len(labels)-1.  With two
        # history-edge tokens, the first returned item is the last history edge
        # and the true candidate consequently has index one.
        labels = np.asarray(observed_ids, dtype=np.int64)
        return {
            "history_ids": [int(value) for value in observed_ids],
            "candidate_ids": [int(value) for value in candidate_ids],
            "history_points": torch.from_numpy(history_points.astype(np.float32)),
            "candidate_points": torch.from_numpy(candidate_points.astype(np.float32)),
            "history_images": torch.from_numpy(history_images.astype(np.float32)),
            "candidate_images": torch.from_numpy(candidate_images.astype(np.float32)),
            "coordinates": torch.from_numpy(coordinates.astype(np.float32)),
            "labels": torch.from_numpy(labels),
            "history_length": len(observed_ids),
            "sequence_id": int(record_index),
            "gt_index": 1,
        }


class PaperEvaluationDataset(TracingDataset):
    """Evaluation states used for the TraceFormer/Neuron-DPO comparison.

    Both released checkpoints are evaluated on exactly the same states.  This
    preserves the original paper code's convention: every valid history length
    is included, the annotated continuation edge appears once as the last
    history token and once as the first candidate token, and its prediction
    index is consequently one.
    """

    def __init__(self, sequence_path: str | Path, point_path: str | Path, npoint: int = 256):
        super().__init__(sequence_path, point_path, npoint)
        self.states = []
        for record_index, record in enumerate(self.records):
            segment_count = len(record[0])
            for history_length in range(1, segment_count):
                candidate_index = history_length - 1
                if (
                    candidate_index < len(record[2])
                    and candidate_index < len(record[4])
                    and len(record[2][candidate_index]) > 0
                    and len(record[4][candidate_index]) > 0
                ):
                    self.states.append((record_index, history_length))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record_index, history_length = self.states[index]
        record = self.records[record_index]

        full_sequence = list(record[0][: history_length + 1])
        observed_ids = full_sequence[:-1]
        target_id = full_sequence[-1]
        negative_ids = list(record[2][history_length - 1])
        candidate_ids = [target_id] + negative_ids

        history_coordinates = np.asarray(record[1][:history_length], dtype=np.float32)
        center = np.mean(history_coordinates, axis=0)

        history_points = [
            self._edge_points(full_sequence[i - 1], full_sequence[i])
            for i in range(1, len(full_sequence))
        ]
        source_id = full_sequence[-2]
        candidate_points = [history_points[-1]] + [
            self._edge_points(source_id, candidate_id) for candidate_id in negative_ids
        ]

        all_points = np.concatenate(history_points + candidate_points, axis=0)
        scale = float(np.max(np.linalg.norm(all_points[:, :3] - center, axis=1)))
        scale = max(scale, 1e-12)
        history_points = (np.stack(history_points) - center) / scale
        candidate_points = (np.stack(candidate_points) - center) / scale
        normalized_coordinates = (history_coordinates - center) / scale

        candidate_image_raw = record[3][history_length - 1]
        positive_image_raw = record[4][history_length - 1]
        negative_images = np.concatenate(
            [np.asarray(item).reshape(1, -1) for item in candidate_image_raw], axis=0
        )
        positive_images = np.concatenate(
            [np.asarray(item).reshape(1, -1) for item in positive_image_raw], axis=0
        )
        combined_images = np.concatenate([positive_images, negative_images], axis=0)
        candidate_count = len(candidate_ids)
        history_images = combined_images[:-candidate_count]
        candidate_images = combined_images[-candidate_count:]

        # The SFT checkpoint projects individual 128-D node features, whereas
        # the Neuron-DPO checkpoint projects a 256-D source/target pair.  Both
        # views are derived from the same node embeddings and correspond to
        # exactly the same history and candidate edges.
        edge_history_images = np.stack(
            [
                np.concatenate([positive_images[i - 1], positive_images[i]])
                for i in range(1, len(positive_images))
            ]
        )
        source_image = positive_images[-2]
        edge_candidate_images = np.stack(
            [np.concatenate([source_image, image]) for image in candidate_images]
        )

        return {
            "history_ids": [int(value) for value in observed_ids],
            "candidate_ids": [int(value) for value in candidate_ids],
            "history_points": torch.from_numpy(history_points.astype(np.float32)),
            "candidate_points": torch.from_numpy(candidate_points.astype(np.float32)),
            "history_images": torch.from_numpy(history_images.astype(np.float32)),
            "candidate_images": torch.from_numpy(candidate_images.astype(np.float32)),
            "edge_history_images": torch.from_numpy(
                edge_history_images.astype(np.float32)
            ),
            "edge_candidate_images": torch.from_numpy(
                edge_candidate_images.astype(np.float32)
            ),
            "coordinates": torch.from_numpy(normalized_coordinates.astype(np.float32)),
            "labels": torch.from_numpy(np.asarray(observed_ids, dtype=np.int64)),
            "history_length": int(history_length),
            "sequence_id": int(record_index),
            "gt_index": 1,
        }
