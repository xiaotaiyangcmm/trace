"""Neuron-DPO training with random or optional checkpoint initialization."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from common import (
    build_from_checkpoint,
    build_from_scratch,
    cleanup_distributed,
    clone_modules,
    encode_state,
    freeze,
    forward_scores,
    init_distributed,
    resolve_inside_project,
    save_checkpoint,
    set_seed,
)
from dataset import TracingDataset, single_state_collate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sequences", default="data/train/chunk_0.pkl")
    parser.add_argument("--train-points", default="data/train/chunk_0_points.pkl")
    parser.add_argument("--model-config", default="config/traceformer.json")
    parser.add_argument(
        "--pretrained-checkpoint",
        default="",
        help="Optional package-local checkpoint; omit to initialize randomly",
    )
    parser.add_argument("--output-dir", default="checkpoints/trained_neurondpo")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--margin-floor", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="Optional smoke-test limit")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--add-image-feature", dest="add_image_feature", action="store_true")
    parser.add_argument("--no-image-feature", dest="add_image_feature", action="store_false")
    parser.set_defaults(add_image_feature=True)
    return parser.parse_args()


def positive_indices(batch):
    positives = [int(batch["gt_index"])]
    negatives = [index for index in range(len(batch["candidate_ids"])) if index not in positives]
    return positives, negatives


def neurondpo_loss(policy_scores, reference_scores, positives, negatives, beta, margin_floor, epsilon):
    if not positives or not negatives:
        return None
    positive = torch.as_tensor(positives, device=policy_scores.device, dtype=torch.long)
    negative = torch.as_tensor(negatives, device=policy_scores.device, dtype=torch.long)

    policy_log_prob = F.log_softmax(policy_scores, dim=0)
    reference_log_prob = F.log_softmax(reference_scores, dim=0)
    log_ratio = policy_log_prob - reference_log_prob
    delta = log_ratio[positive][:, None] - log_ratio[negative][None, :]

    # The coefficient and its normalization are constants during backward.
    weights = delta.detach().abs().clamp_min(margin_floor)
    pair_loss = -F.logsigmoid(beta * delta)
    return (weights * pair_loss).sum() / (weights.sum() + epsilon)


def main():
    args = parse_args()
    rank, world_size, local_rank, device = init_distributed()
    set_seed(args.seed, rank)

    train_sequences = resolve_inside_project(args.train_sequences)
    train_points = resolve_inside_project(args.train_points)
    model_config = resolve_inside_project(args.model_config)
    pretrained_checkpoint = (
        resolve_inside_project(args.pretrained_checkpoint)
        if args.pretrained_checkpoint
        else None
    )
    output_dir = resolve_inside_project(args.output_dir)

    dataset = TracingDataset(train_sequences, train_points)
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=single_state_collate,
        pin_memory=True,
    )
    if pretrained_checkpoint is None:
        policy, encoder, image_mlp = build_from_scratch(model_config, device)
        reference, reference_encoder, reference_image_mlp = clone_modules(
            policy, encoder, image_mlp
        )
        train_feature_modules = True
        initialization = "random"
    else:
        policy, encoder, image_mlp = build_from_checkpoint(pretrained_checkpoint, device)
        reference, reference_encoder, reference_image_mlp = build_from_checkpoint(
            pretrained_checkpoint, device
        )
        # Preserve the lightweight fine-tuning behavior when users supply a
        # fully trained feature checkpoint.
        freeze(encoder)
        freeze(image_mlp)
        train_feature_modules = False
        initialization = str(pretrained_checkpoint)

    freeze(reference)
    freeze(reference_encoder)
    freeze(reference_image_mlp)
    policy.train()
    if train_feature_modules:
        encoder.train()
        image_mlp.train()

    if rank == 0:
        print(f"Initialization: {initialization}", flush=True)

    if world_size > 1:
        # TraceFormer consumes continuous feature embeddings directly, so its
        # token-embedding table is intentionally unused in Neuron-DPO.  Tell
        # DDP to exclude such parameters from gradient synchronization.
        policy = DDP(
            policy,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        if train_feature_modules:
            encoder = DDP(
                encoder,
                device_ids=[local_rank],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
            image_mlp = DDP(
                image_mlp, device_ids=[local_rank], broadcast_buffers=False
            )

    trainable_parameters = list(
        parameter
        for parameter in itertools.chain(
            policy.parameters(), encoder.parameters(), image_mlp.parameters()
        )
        if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation)
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_updates = int(total_updates * args.warmup_ratio)

    def lr_lambda(step):
        if warmup_updates and step < warmup_updates:
            return float(step + 1) / float(warmup_updates)
        progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    optimizer_step = 0

    stop_training = False
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        running_loss = 0.0
        valid_steps = 0

        for batch in loader:
            global_step += 1
            with torch.no_grad():
                reference_encoded = encode_state(
                    batch,
                    reference_encoder,
                    reference_image_mlp,
                    device,
                    args.add_image_feature,
                )
                _, _, reference_scores = forward_scores(reference, reference_encoded)
            policy_encoded = encode_state(
                batch, encoder, image_mlp, device, args.add_image_feature
            )
            _, _, policy_scores = forward_scores(policy, policy_encoded)
            positives, negatives = positive_indices(batch)
            loss = neurondpo_loss(
                policy_scores,
                reference_scores,
                positives,
                negatives,
                args.beta,
                args.margin_floor,
                args.epsilon,
            )
            if loss is None:
                continue

            (loss / args.gradient_accumulation).backward()
            running_loss += float(loss.detach())
            valid_steps += 1

            if valid_steps % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            if rank == 0 and global_step % args.log_every == 0:
                mean_loss = running_loss / max(1, valid_steps)
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={global_step} "
                    f"updates={optimizer_step} loss={mean_loss:.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )

            if rank == 0 and args.save_every > 0 and global_step % args.save_every == 0:
                save_checkpoint(output_dir / f"step_{global_step}", policy, encoder, image_mlp)

            if args.max_steps > 0 and global_step >= args.max_steps:
                stop_training = True
                break

        if valid_steps % args.gradient_accumulation:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

        if rank == 0:
            save_checkpoint(output_dir / f"epoch_{epoch + 1}", policy, encoder, image_mlp)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if stop_training:
            break

    if rank == 0:
        save_checkpoint(output_dir / "final", policy, encoder, image_mlp)
        print(f"Training complete. Final checkpoint: {output_dir / 'final'}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
