"""Evaluate TraceFormer with or without Neuron-DPO on one shared test set."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, Sampler, Subset
from tqdm import tqdm

from common import (
    all_reduce_sum,
    build_from_checkpoint,
    cleanup_distributed,
    encode_state,
    forward_scores,
    freeze,
    init_distributed,
    resolve_inside_project,
    set_seed,
)
from dataset import PaperEvaluationDataset, TracingDataset, single_state_collate


class ExactDistributedSampler(Sampler):
    """Shard evaluation without duplicating samples for padding."""

    def __init__(self, dataset, rank: int, world_size: int):
        self.indices = list(range(rank, len(dataset), world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("traceformer", "neurondpo"),
        default="neurondpo",
        help="Use the SFT TraceFormer or the Neuron-DPO checkpoint.",
    )
    parser.add_argument(
        "--state-protocol",
        choices=("auto", "traceformer", "neurondpo"),
        default="auto",
        help="State construction protocol; auto selects the released protocol for each method.",
    )
    parser.add_argument("--checkpoint", default="", help="Optional package-local override")
    parser.add_argument("--test-sequences", default="data/test/chunk_1.pkl")
    parser.add_argument("--test-points", default="data/test/chunk_1_points.pkl")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-states", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--add-image-feature", dest="add_image_feature", action="store_true")
    parser.add_argument("--no-image-feature", dest="add_image_feature", action="store_false")
    parser.set_defaults(add_image_feature=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, _, device = init_distributed()
    set_seed(args.seed, rank)

    checkpoint = args.checkpoint or f"checkpoints/{args.method}"
    state_protocol = args.method if args.state_protocol == "auto" else args.state_protocol
    dataset_class = PaperEvaluationDataset if state_protocol == "traceformer" else TracingDataset
    dataset = dataset_class(
        resolve_inside_project(args.test_sequences),
        resolve_inside_project(args.test_points),
    )
    if args.max_states > 0:
        dataset = Subset(dataset, range(min(args.max_states, len(dataset))))
    sampler = ExactDistributedSampler(dataset, rank, world_size) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=single_state_collate,
        pin_memory=True,
    )

    model, encoder, image_mlp = build_from_checkpoint(
        resolve_inside_project(checkpoint), device
    )
    freeze(model)
    freeze(encoder)
    freeze(image_mlp)

    count = 0.0
    top1 = 0.0
    top5 = 0.0
    rank_sum = 0.0
    iterator = tqdm(
        loader,
        disable=rank != 0,
        desc=f"Evaluating {args.method}",
        bar_format="{desc}: {percentage:3.0f}%",
    )
    with torch.inference_mode():
        for batch in iterator:
            if args.method == "traceformer" and state_protocol == "neurondpo":
                batch = dict(batch)
                batch["history_images"] = batch["traceformer_history_images"]
                batch["candidate_images"] = batch["traceformer_candidate_images"]
            elif args.method == "neurondpo" and state_protocol == "traceformer":
                batch = dict(batch)
                batch["history_images"] = batch["edge_history_images"]
                batch["candidate_images"] = batch["edge_candidate_images"]
            encoded = encode_state(batch, encoder, image_mlp, device, args.add_image_feature)
            _, predictions, _ = forward_scores(model, encoded)
            ordering = torch.argsort(predictions, descending=True)
            gt_rank_zero = int(
                torch.nonzero(ordering == int(batch["gt_index"]), as_tuple=False)[0]
            )
            count += 1
            top1 += float(gt_rank_zero == 0)
            top5 += float(gt_rank_zero < min(5, len(ordering)))
            rank_sum += gt_rank_zero + 1

    total_count, total_top1, total_top5, total_rank = all_reduce_sum(
        [count, top1, top5, rank_sum], device
    ).cpu().tolist()
    if rank == 0:
        print("-" * 72)
        print(f"Method:     {args.method}")
        print(f"Top-1:      {total_top1 / total_count:.10f}")
        print(f"Top-5:      {total_top5 / total_count:.10f}")
        print(f"MRank:      {total_rank / total_count:.10f}")
        print("-" * 72)
    cleanup_distributed()


if __name__ == "__main__":
    main()
