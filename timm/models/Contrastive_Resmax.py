import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
from timm.models._registry import register_model
from .RESMAX import RESMAX_V2

# === Shared Loss Functions ===
def nt_xent_loss(f1, f2, temperature):
    batch_size = f1.size(0)
    if len(f1.shape) > 2:
        f1 = f1.reshape(f1.size(0), -1)
        f2 = f2.reshape(f2.size(0), -1)
    z1 = F.normalize(f1, dim=1)
    z2 = F.normalize(f2, dim=1)
    features = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.matmul(features, features.T) / temperature
    pos_mask = torch.zeros_like(sim_matrix)
    pos_mask[:batch_size, batch_size:] = torch.eye(batch_size)
    pos_mask[batch_size:, :batch_size] = torch.eye(batch_size)
    self_mask = torch.eye(2 * batch_size, device=sim_matrix.device)
    logits_mask = torch.ones_like(sim_matrix) - self_mask
    exp_logits = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True))
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
    return -mean_log_prob_pos.mean()

def calc_contrastive_loss(f1_list, f2_list, temperature):
    if isinstance(f1_list, list):
        total = sum(nt_xent_loss(f1, f2, temperature) for f1, f2 in zip(f1_list, f2_list))
        return total / len(f1_list)
    else:
        return nt_xent_loss(f1_list, f2_list, temperature)

def clip_style_loss(z1, z2, temperature):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits_per_orig = torch.matmul(z1, z2.T) / temperature
    logits_per_scaled = torch.matmul(z2, z1.T) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_orig = F.cross_entropy(logits_per_orig, labels)
    loss_scaled = F.cross_entropy(logits_per_scaled, labels)
    return (loss_orig + loss_scaled) / 2

# === Contrastive Model Classes ===

class ContrastiveRESMAX(nn.Module):
    def __init__(self,
                num_classes=1000,
                in_chans=3,
                ip_scale_bands=1,
                classifier_input_size=9216,
                contrastive_loss=True,
                bypass=False,
                pretrained_path=None,
                temperature=0.1,
                **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.contrastive_loss = contrastive_loss
        self.temperature = temperature
        self.ip_scale_bands = ip_scale_bands
        self.backbone = RESMAX_V2(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )
        # Load pretrained weights if path is given
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, weights_only=False, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.backbone.load_state_dict(checkpoint['state_dict'], strict=True)
            else:
                self.backbone.load_state_dict(checkpoint, strict=True)
        # Freeze all backbone params
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Unfreeze FC and S3, global_pool
        for param in self.backbone.fc2.parameters():
            param.requires_grad = True
        for param in self.backbone.fc1.parameters():
            param.requires_grad = True
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
        for param in self.backbone.s3.parameters():
            param.requires_grad = True
        for param in self.backbone.global_pool.parameters():
            param.requires_grad = True
    def forward(self, x):
        batch_size = x.shape[0]
        if self.backbone.bypass:
            out1, c1_feats1, c2_feats1, bypass1 = self.backbone(x)
        else:
            out1, c1_feats1, c2_feats1 = self.backbone(x)
        # Randomly scaled input
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)
        if new_hw <= img_hw:
            x_rescaled = F.pad(x_rescaled, (0, img_hw - new_hw, 0, img_hw - new_hw))
        else:
            center_crop = transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)
        if self.backbone.bypass:
            out2, c1_feats2, c2_feats2, bypass2 = self.backbone(x_rescaled)
        else:
            out2, c1_feats2, c2_feats2 = self.backbone(x_rescaled)
        c1_contrastive_loss = calc_contrastive_loss(c1_feats1, c1_feats2, self.temperature)
        c2_contrastive_loss = calc_contrastive_loss(c2_feats1, c2_feats2, self.temperature)
        out_contrastive_loss = nt_xent_loss(out1, out2, self.temperature)
        if self.backbone.bypass:
            bypass_contrastive_loss = nt_xent_loss(bypass1, bypass2, self.temperature)
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss + 0.1 * bypass_contrastive_loss
        else:
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss
        return out1, total_loss

class ContrastiveRESMAX_V1(nn.Module):
    def __init__(self,
                num_classes=1000,
                in_chans=3,
                ip_scale_bands=1,
                classifier_input_size=9216,
                contrastive_loss=True,
                bypass=False,
                pretrained_path=None,
                temperature=0.1,
                **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.contrastive_loss = contrastive_loss
        self.temperature = temperature
        self.ip_scale_bands = ip_scale_bands
        self.backbone = RESMAX_V2(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, weights_only=False, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.backbone.load_state_dict(checkpoint['state_dict'], strict=True)
            else:
                self.backbone.load_state_dict(checkpoint, strict=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc2.parameters():
            param.requires_grad = True
        for param in self.backbone.fc1.parameters():
            param.requires_grad = True
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
        for param in self.backbone.s3.parameters():
            param.requires_grad = True
        for param in self.backbone.global_pool.parameters():
            param.requires_grad = True
    def forward(self, x):
        batch_size = x.shape[0]
        if self.backbone.bypass:
            out1, c1_feats1, c2_feats1, bypass1 = self.backbone(x)
        else:
            out1, c1_feats1, c2_feats1 = self.backbone(x)
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)
        if new_hw <= img_hw:
            x_rescaled = F.pad(x_rescaled, (0, img_hw - new_hw, 0, img_hw - new_hw))
        else:
            center_crop = transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)
        if self.backbone.bypass:
            out2, c1_feats2, c2_feats2, bypass2 = self.backbone(x_rescaled)
        else:
            out2, c1_feats2, c2_feats2 = self.backbone(x_rescaled)
        c1_contrastive_loss = calc_contrastive_loss(c1_feats1, c1_feats2, self.temperature)
        c2_contrastive_loss = calc_contrastive_loss(c2_feats1, c2_feats2, self.temperature)
        out_contrastive_loss = clip_style_loss(out1, out2, self.temperature)
        if self.backbone.bypass:
            bypass_contrastive_loss = clip_style_loss(bypass1, bypass2, self.temperature)
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss + 0.1 * bypass_contrastive_loss
        else:
            total_loss = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss
        return out1, total_loss

class ContrastiveRESMAX_V2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 in_chans=3,
                 ip_scale_bands=1,
                 classifier_input_size=9216,
                 contrastive_loss=True,
                 bypass=False,
                 pretrained_path=None,
                 temperature=0.1,
                 use_kl_loss=True,
                 kl_loss_weight=0.5,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.contrastive_loss = contrastive_loss
        self.temperature = temperature
        self.use_kl_loss = use_kl_loss
        self.kl_loss_weight = kl_loss_weight
        self.ip_scale_bands = ip_scale_bands
        self.backbone = RESMAX_V2(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, weights_only=False, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.backbone.load_state_dict(checkpoint['state_dict'], strict=True)
            else:
                self.backbone.load_state_dict(checkpoint, strict=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
        for param in self.backbone.fc1.parameters():
            param.requires_grad = True
        for param in self.backbone.fc2.parameters():
            param.requires_grad = True
        for param in self.backbone.s3.parameters():
            param.requires_grad = True
        for param in self.backbone.global_pool.parameters():
            param.requires_grad = True
    def forward(self, x):
        batch_size = x.shape[0]
        with torch.no_grad():
            teacher_out, *_ = self.backbone(x)
            teacher_log_probs = F.log_softmax(teacher_out / self.temperature, dim=1)
        if self.backbone.bypass:
            out1, c1_feats1, c2_feats1, bypass1 = self.backbone(x)
        else:
            out1, c1_feats1, c2_feats1 = self.backbone(x)
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)
        if new_hw <= img_hw:
            x_rescaled = F.pad(x_rescaled, (0, img_hw - new_hw, 0, img_hw - new_hw))
        else:
            center_crop = transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)
        if self.backbone.bypass:
            out2, c1_feats2, c2_feats2, bypass2 = self.backbone(x_rescaled)
        else:
            out2, c1_feats2, c2_feats2 = self.backbone(x_rescaled)
        c1_contrastive_loss = calc_contrastive_loss(c1_feats1, c1_feats2, self.temperature)
        c2_contrastive_loss = calc_contrastive_loss(c2_feats1, c2_feats2, self.temperature)
        out_contrastive_loss = nt_xent_loss(out1, out2, self.temperature)
        if self.backbone.bypass:
            bypass_contrastive_loss = nt_xent_loss(bypass1, bypass2, self.temperature)
            contrastive_total = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss + 0.1 * bypass_contrastive_loss
        else:
            contrastive_total = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss
        if self.use_kl_loss:
            student_log_probs = F.log_softmax(out1 / self.temperature, dim=1)
            teacher_probs = F.softmax(teacher_out / self.temperature, dim=1)
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)
            total_loss = contrastive_total + self.kl_loss_weight * kl_loss
        else:
            total_loss = contrastive_total
        return out1, total_loss

class ContrastiveRESMAX_V3(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 in_chans=3,
                 ip_scale_bands=1,
                 classifier_input_size=9216,
                 contrastive_loss=True,
                 bypass=False,
                 pretrained_path=None,
                 temperature=0.1,
                 use_kl_loss=True,
                 kl_loss_weight=0.5,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.contrastive_loss = contrastive_loss
        self.temperature = temperature
        self.use_kl_loss = use_kl_loss
        self.kl_loss_weight = kl_loss_weight
        self.ip_scale_bands = ip_scale_bands
        self.teacher = RESMAX_V2(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )
        pretrained_path='/oscar/data/tserre/xyu110/pytorch-output/train/0/models_w_aug/ip_3_resmax_v2_gpu_8_cl_0_ip_3_322_322_18432_c1[_6,3,1_]_bypass_scale_0.08/model_best.pth.tar'
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, weights_only=False, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.teacher.load_state_dict(checkpoint['state_dict'], strict=True)
            else:
                self.teacher.load_state_dict(checkpoint, strict=True)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        self.student = RESMAX_V2(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=self.ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=self.contrastive_loss,
            bypass=bypass,
        )
        self.student.load_state_dict(self.teacher.state_dict(), strict=True)
        for param in self.student.parameters():
            param.requires_grad = False
        for param in self.student.c1.parameters():
            param.requires_grad = True
        for param in self.student.c2.parameters():
            param.requires_grad = True
        for param in self.student.c2b_seq.parameters():
            param.requires_grad = True
        for param in self.student.global_pool.parameters():
            param.requires_grad = True
    def forward(self, x, train=True):
        batch_size = x.shape[0]
        if train:
            with torch.no_grad():
                teacher_out, *_ = self.teacher(x)
                teacher_log_probs = F.log_softmax(teacher_out / self.temperature, dim=1)
        if self.student.bypass:
            out1, c1_feats1, c2_feats1, bypass1 = self.student(x)
        else:
            out1, c1_feats1, c2_feats1 = self.student(x)
        if not train:
            return out1
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)
        if new_hw <= img_hw:
            x_rescaled = F.pad(x_rescaled, (0, img_hw - new_hw, 0, img_hw - new_hw))
        else:
            center_crop = transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)
        if self.student.bypass:
            out2, c1_feats2, c2_feats2, bypass2 = self.student(x_rescaled)
        else:
            out2, c1_feats2, c2_feats2 = self.student(x_rescaled)
        c1_contrastive_loss = calc_contrastive_loss(c1_feats1, c1_feats2, self.temperature)
        c2_contrastive_loss = calc_contrastive_loss(c2_feats1, c2_feats2, self.temperature)
        out_contrastive_loss = nt_xent_loss(out1, out2, self.temperature)
        if self.student.bypass:
            bypass_contrastive_loss = nt_xent_loss(bypass1, bypass2, self.temperature)
            contrastive_total = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss + 0.1 * bypass_contrastive_loss
        else:
            contrastive_total = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss
        if self.use_kl_loss:
            student_log_probs = F.log_softmax(out1 / self.temperature, dim=1)
            teacher_probs = F.softmax(teacher_out / self.temperature, dim=1)
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)
            total_loss = contrastive_total + self.kl_loss_weight * kl_loss
        else:
            total_loss = contrastive_total
        return out1, total_loss

class ContrastiveRESMAX_V4(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 in_chans=3,
                 teacher_ip_scale_bands=3,
                 student_ip_scale_bands=7,
                 classifier_input_size=9216,
                 contrastive_loss=True,
                 bypass=False,
                 pretrained_path=None,
                 temperature=0.1,
                 use_kl_loss=True,
                 kl_loss_weight=0.5,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.contrastive_loss = contrastive_loss
        self.temperature = temperature
        self.use_kl_loss = use_kl_loss
        self.kl_loss_weight = kl_loss_weight

        # Teacher model (frozen)
        self.teacher = RESMAX_V2(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=teacher_ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=contrastive_loss,
            bypass=bypass,
        )
        pretrained_path='/oscar/data/tserre/xyu110/pytorch-output/train/0/models_w_aug/ip_3_resmax_v2_gpu_8_cl_0_ip_3_322_322_18432_c1[_6,3,1_]_bypass_scale_0.08/model_best.pth.tar'
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, weights_only=False, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.teacher.load_state_dict(checkpoint['state_dict'], strict=True)
            else:
                self.teacher.load_state_dict(checkpoint, strict=True)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # Student model (trainable, with more scale bands)
        self.student = RESMAX_V2(
            num_classes=num_classes,
            in_chans=in_chans,
            ip_scale_bands=student_ip_scale_bands,
            classifier_input_size=classifier_input_size,
            contrastive_loss=contrastive_loss,
            bypass=bypass,
        )
        self.student.load_state_dict(self.teacher.state_dict(), strict=False)
        for param in self.student.parameters():
            param.requires_grad = False
        for param in self.student.c1.parameters():
            param.requires_grad = True
        for param in self.student.c2.parameters():
            param.requires_grad = True
        for param in self.student.c2b_seq.parameters():
            param.requires_grad = True
        for param in self.student.global_pool.parameters():
            param.requires_grad = True

    def forward(self, x, train=True):
        batch_size = x.shape[0]
        if train:
            with torch.no_grad():
                teacher_out, *_ = self.teacher(x)
        if self.student.bypass:
            out1, c1_feats1, c2_feats1, bypass1 = self.student(x)
        else:
            out1, c1_feats1, c2_feats1 = self.student(x)
        if not train:
            return out1
        scale_factor_list = [0.49, 0.59, 0.707, 0.841, 1.0, 1.189, 1.414, 1.681, 2.0]
        scale_factor = random.choice(scale_factor_list)
        img_hw = x.shape[-1]
        new_hw = int(img_hw * scale_factor)
        x_rescaled = F.interpolate(x, size=(new_hw, new_hw), mode='bilinear', align_corners=False)
        if new_hw <= img_hw:
            x_rescaled = F.pad(x_rescaled, (0, img_hw - new_hw, 0, img_hw - new_hw))
        else:
            center_crop = transforms.CenterCrop(img_hw)
            x_rescaled = center_crop(x_rescaled)
        if self.student.bypass:
            out2, c1_feats2, c2_feats2, bypass2 = self.student(x_rescaled)
        else:
            out2, c1_feats2, c2_feats2 = self.student(x_rescaled)
        c1_contrastive_loss = calc_contrastive_loss(c1_feats1, c1_feats2, self.temperature)
        c2_contrastive_loss = calc_contrastive_loss(c2_feats1, c2_feats2, self.temperature)
        out_contrastive_loss = nt_xent_loss(out1, out2, self.temperature)
        if self.student.bypass:
            bypass_contrastive_loss = nt_xent_loss(bypass1, bypass2, self.temperature)
            contrastive_total = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss + 0.1 * bypass_contrastive_loss
        else:
            contrastive_total = c1_contrastive_loss + c2_contrastive_loss + 0.1 * out_contrastive_loss
        if self.use_kl_loss:
            student_log_probs = F.log_softmax(out1 / self.temperature, dim=1)
            teacher_probs = F.softmax(teacher_out / self.temperature, dim=1)
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)
            total_loss = contrastive_total + self.kl_loss_weight * kl_loss
        else:
            total_loss = contrastive_total
        return out2, total_loss

# === Model Registration ===
@register_model
def contrastive_resmax(pretrained=False, **kwargs):
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)
    if pretrained:
        pass
    model = ContrastiveRESMAX(**kwargs)
    return model

@register_model
def contrastive_resmaxv1(pretrained=False, **kwargs):
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)
    if pretrained:
        pass
    model = ContrastiveRESMAX_V1(**kwargs)
    return model

@register_model
def ft_resmax_v2(pretrained=False, **kwargs):
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)
    if pretrained:
        pass
    model = ContrastiveRESMAX_V2(**kwargs)
    return model

@register_model
def ft_resmax_v3(pretrained=False, **kwargs):
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)
    model = ContrastiveRESMAX_V3(**kwargs)
    return model

@register_model
def ft_resmax_v4(pretrained=False, **kwargs):
    for key in ["pretrained_cfg", "pretrained_cfg_overlay", "drop_rate"]:
        kwargs.pop(key, None)
    model = ContrastiveRESMAX_V4(**kwargs)
    return model
