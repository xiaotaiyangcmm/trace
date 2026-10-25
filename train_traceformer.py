"""Supervised TraceFormer training without Neuron-DPO."""

from __future__ import annotations

import argparse
import itertools
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from common import (
    build_from_checkpoint,
    build_from_scratch,
    cleanup_distributed,
    encode_state,
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
    parser.add_argument("--output-dir", default="checkpoints/trained_traceformer")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
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
        model, encoder, image_mlp = build_from_scratch(model_config, device)
        initialization = "random"
    else:
        model, encoder, image_mlp = build_from_checkpoint(pretrained_checkpoint, device)
        initialization = str(pretrained_checkpoint)

    model.train()
    encoder.train()
    image_mlp.train()
    if rank == 0:
        print(f"Initialization: {initialization}", flush=True)

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        encoder = DDP(
            encoder,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        image_mlp = DDP(image_mlp, device_ids=[local_rank], broadcast_buffers=False)

    trainable_parameters = list(
        itertools.chain(model.parameters(), encoder.parameters(), image_mlp.parameters())
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
            encoded = encode_state(
                batch, encoder, image_mlp, device, args.add_image_feature
            )
            _, _, scores = forward_scores(model, encoded)
            target = torch.tensor([batch["gt_index"]], dtype=torch.long, device=device)
            loss = F.cross_entropy(scores.unsqueeze(0), target)

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
                save_checkpoint(output_dir / f"step_{global_step}", model, encoder, image_mlp)

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
            save_checkpoint(output_dir / f"epoch_{epoch + 1}", model, encoder, image_mlp)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if stop_training:
            break

    if rank == 0:
        save_checkpoint(output_dir / "final", model, encoder, image_mlp)
        print(f"Training complete. Final checkpoint: {output_dir / 'final'}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
