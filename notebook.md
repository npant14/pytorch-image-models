---
noteId: "10c800902dc711f0a61ff7cbd563f22c"
tags: []

---

## ContrastiveRESMAX_V3: Teacher-Student Contrastive Distillation Model

This model implements a teacher-student architecture for contrastive learning and knowledge distillation.

- **Teacher**: A frozen RESMAX_V2 model loaded with pretrained weights.
- **Student**: Initialized with the same weights as the teacher, but only the C1, C2, C2b, and global_pool layers are unfrozen for fine-tuning.
- Both teacher and student use the same number of image pyramid scale bands (`ip_scale_bands`).

### Training
- The teacher provides stable, pretrained outputs for knowledge distillation (KL loss).
- The student is trained with a combination of contrastive loss (NT-Xent) between original and randomly-rescaled images, and KL divergence loss to match the teacher's output.
- Only the specified layers of the student are updated during training; all other layers remain frozen.

### Arguments
- `num_classes (int)`: Number of output classes.
- `in_chans (int)`: Number of input channels.
- `ip_scale_bands (int)`: Number of image pyramid scale bands.
- `classifier_input_size (int)`: Input size for the classifier head.
- `contrastive_loss (bool)`: Whether to use contrastive loss.
- `bypass (bool)`: Whether to use the bypass path in the backbone.
- `pretrained_path (str)`: Path to the pretrained weights for the teacher.
- `temperature (float)`: Temperature for contrastive and KL losses.
- `use_kl_loss (bool)`: Whether to use KL divergence loss.
- `kl_loss_weight (float)`: Weight for the KL divergence loss.
- `**kwargs`: Additional arguments for the backbone.

### Forward
- `x (Tensor)`: Input batch of images.
- `train (bool)`: If True, computes both contrastive and KL losses. If False, only returns student output.

### Returns
- `out1 (Tensor)`: Student model output.
- `total_loss (Tensor or 0)`: Combined loss (if train=True), or 0 (if train=False).

## ContrastiveRESMAX_V4: Teacher-Student with Flexible Student Scale Bands

This model extends the teacher-student contrastive distillation approach by allowing the student to use a different (typically larger) number of image pyramid scale bands than the teacher.

- **Teacher**: A frozen RESMAX_V2 model loaded with pretrained weights, using `teacher_ip_scale_bands` (default: 3).
- **Student**: Initialized with the same weights as the teacher, but uses `student_ip_scale_bands` (default: 7, configurable). Only the C1, C2, C2b, and global_pool layers are unfrozen for fine-tuning.
- This flexibility allows the student to learn from more scales, potentially improving robustness and generalization.

### Training
- The teacher provides stable, pretrained outputs for knowledge distillation (KL loss).
- The student is trained with a combination of contrastive loss (NT-Xent) between original and randomly-rescaled images, and KL divergence loss to match the teacher's output.
- Only the specified layers of the student are updated during training; all other layers remain frozen.

### Arguments
- `num_classes (int)`: Number of output classes.
- `in_chans (int)`: Number of input channels.
- `teacher_ip_scale_bands (int)`: Number of image pyramid scale bands for the teacher (default: 3).
- `student_ip_scale_bands (int)`: Number of image pyramid scale bands for the student (default: 7).
- `classifier_input_size (int)`: Input size for the classifier head.
- `contrastive_loss (bool)`: Whether to use contrastive loss.
- `bypass (bool)`: Whether to use the bypass path in the backbone.
- `pretrained_path (str)`: Path to the pretrained weights for the teacher.
- `temperature (float)`: Temperature for contrastive and KL losses.
- `use_kl_loss (bool)`: Whether to use KL divergence loss.
- `kl_loss_weight (float)`: Weight for the KL divergence loss.
- `**kwargs`: Additional arguments for the backbone.

### Forward
- `x (Tensor)`: Input batch of images.
- `train (bool)`: If True, computes both contrastive and KL losses. If False, only returns student output.

### Returns
- `out1 (Tensor)`: Student model output.
- `out2 (Tensor)`: Teacher model output.
- `total_loss (Tensor or 0)`: Combined loss (if train=True), or 0 (if train=False).

## Update: ContrastiveRESMAX_V4 Now Returns Both Student and Teacher Outputs

**Change:**
- The `ContrastiveRESMAX_V4` model now returns both the student and teacher outputs from its forward pass.

### New Return Signature
- `out1 (Tensor)`: Student model output (fine-tuned, typically with more scale bands).
- `out2 (Tensor)`: Teacher model output (frozen, typically with fewer scale bands).
- `total_loss (Tensor or 0)`: Combined loss (if `train=True`), or 0 (if `train=False`).

### Implications
- This change allows downstream code to access both the student and teacher predictions for further analysis, visualization, or custom loss computation.
- When using the model in training or evaluation, ensure your code is updated to handle the new tuple output: `(out1, out2, total_loss)`.

### Example Usage
```python
out1, out2, total_loss = model(x, train=True)
```

- `out1` is used for student evaluation and loss computation.
- `out2` can be used for teacher-student comparison, distillation, or interpretability tasks.

