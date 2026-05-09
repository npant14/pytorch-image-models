## Figure 2: Scale Invariance on ImageNet-1k

This repository includes the evaluation setup used for Figure 2 (scale invariance on ImageNet-1k validation).

The multi-scale validation entrypoint is:

```bash
./imagenet/val_job.sh <model> 0 0 0 1 128 False "160 192 227 270 322 382 454" "1:00:00" ""

./imagenet/validation_top1_dict.py --csv validation_imagenet.csv
```

### Model choices and checkpoints used

- `alexnet_timm` -> `alexnet` with pretrained weights from:
  `https://download.pytorch.org/models/alexnet-owt-7be5be79.pth`
- `resnet18_timm` -> `resnet18.tv_in1k` (torchvision ResNet-18 weights via timm):
  `https://download.pytorch.org/models/resnet18-5c106cde.pth`
- `vit_base_patch16_224` -> `vit_base_patch16_224.orig_in21k_ft_in1k`
  (the original ViT-B/16 in21k->in1k variant exposed by timm)
- `hmax_v3_adj` -> local training checkpoint passed through `ckpt_dir` in `val_job.sh`

These aliases are resolved in [`imagenet/val_template.sh`](/users/xyu110/pytorch-image-models/imagenet/val_template.sh).

### Notes

- For `alexnet_timm`, `resnet18_timm`, and ViT models, validation runs with `--pretrained` and no local checkpoint.
- For HMAX models (for example `hmax_v3_adj`), validation uses the checkpoint directory provided to `val_job.sh`.

## Existing HMAX Notes

You can still run any of the models originally included in the library, as well as any of the HMAX models in `timm/models/HMAX.py`.

To run with contrastive loss, use CHMAX as the model and pass:

- `ip_scale_bands`: number of scale bands (one more than number of images in the pyramid)
- `classifier_input_size`: explicit input size to the classifier
- `hmax_type`: currently `"full"` or `"bypass"`

Also set `--cl-lambda <value>` in the run script. Default is `0` (contrastive term disabled).

No other models currently run with contrastive loss.

## Figure 3: Hangul Experiment

This repository also includes the Hangul character evaluation used for Figure 3.

Run:

```bash
sbatch hangul/e0_submit_korean_all_layers.sh
```

Models in the SLURM array job:

- `hmax_v3_adj`
- `resnet18_timm`
- `alexnet_timm`
- `vit_base`
