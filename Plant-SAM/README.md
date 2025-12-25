# Training instruction for Plant-SAM

## 1. Data Preparation

LeafSeg-22K can be downloaded from https://huggingface.co/sam-hq-team/sam-hq-training/tree/main/data)


## 2. Expected checkpoint

```
pretrained_checkpoint
|____sam_vit_b_maskdecoder.pth
|____sam_vit_b_01ec64.pth
|____sam_vit_l_maskdecoder.pth
|____sam_vit_l_0b3195.pth
|____sam_vit_h_maskdecoder.pth
|____sam_vit_h_4b8939.pth

```

## 3. Training
To train Plant-SAM on LeafSeg-22Kdataset

```
python -m torch.distributed.launch --nproc_per_node=<num_gpus> train.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output>
```


## 4. Evaluation
To evaluate on LeafSeg-22K-datasets

```
python -m torch.distributed.launch --nproc_per_node=<num_gpus> train.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output> --eval --restore-model <path/to/training_checkpoint>
```

