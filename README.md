# TraceFormer + Neuron-DPO

Official implementation of **“Long-range Neuron Tracing via Global Context
Modeling and Direct Preference Optimization.”** 

## Environment

Most of the dependencies are included in this Docker image:

```dockerfile
FROM registry.cn-hangzhou.aliyuncs.com/neurondpo/neurondpo:v1
```

## Test / Inference

Download each folder separately from Baidu Netdisk and place it at the path
shown below:

| Folder | Baidu Netdisk | Code |
|:--|:--|:--|
| `data/test/` | `<BAIDU_TEST_DATA_URL>` | `<CODE>` |
| `checkpoints/traceformer/` | `<BAIDU_TRACEFORMER_URL>` | `<CODE>` |
| `checkpoints/neurondpo/` | `<BAIDU_NEURONDPO_URL>` | `<CODE>` |

Run TraceFormer inference without Neuron-DPO:

```bash
torchrun --standalone --nproc_per_node=8 inference.py \
  --method traceformer
```

Run inference with Neuron-DPO:

```bash
torchrun --standalone --nproc_per_node=8 inference.py \
  --method neurondpo
```

## Training

Download the training folder from Baidu Netdisk and place it at the path shown
below:

| Folder | Baidu Netdisk | Code |
|:--|:--|:--|
| `data/train/` | `<BAIDU_TRAIN_DATA_URL>` | `<CODE>` |

Train TraceFormer without Neuron-DPO:

```bash
torchrun --standalone --nproc_per_node=8 train_traceformer.py
```

The trained TraceFormer checkpoint is saved to:

```text
checkpoints/trained_traceformer/final
```

Then run Neuron-DPO training from the trained TraceFormer checkpoint:

```bash
torchrun --standalone --nproc_per_node=8 train_neurondpo.py \
  --pretrained-checkpoint checkpoints/trained_traceformer/final
```

The trained Neuron-DPO checkpoint is saved to:

```text
checkpoints/trained_neurondpo/final
```
