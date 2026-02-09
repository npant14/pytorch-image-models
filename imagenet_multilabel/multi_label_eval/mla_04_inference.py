import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from torch.utils.data import Subset, DataLoader, Dataset
from torchmetrics.regression import SpearmanCorrCoef

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import pandas as pd
import csv

import argparse
import glob
import os
import gc
import time
import warnings
warnings.filterwarnings("ignore")

from dino import DINOv1, DINOv2

def clear_unused_memo(device_name, needClearGC):
    if needClearGC:
        gc.collect()
    
    if device_name:
        with torch.cuda.device(device_name):
            torch.cuda.empty_cache()

def write_csv_all(record, path):
    header = ['model', 'multi_label_acc', 'time']
    file_exists = os.path.isfile(path)

    with open(path, mode='a+', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(record)

class Models:
    def __init__(self):
        self.model_families = [
            'test', 'subset1', 'subset2',
            'beit', 'caformer', 'cait', 'coat', 'coatnet', 
            'convit', 'convmixer', 'davit', 'deit', 'deit3',
            'densenet', 'dla', 'dpn', 'efficientformer', 'efficientnet',
            'focalnet', 'hrnet', 'lcnet', 'levit', 'maxvit', 
            'mixnet', 'mobilenet', 'mvit', 'pit', 'poolformer', 
            'pvt', 'regnet', 'repvgg', 'res2net', 'resnest', 
            'resnet', 'rexnet', 'sknet', 'swin', 'tresnet', 
            'twins', 'vgg', 'visformer', 'vit_base', 'vit', 'vit_large',
            'vit_relpos', 'volo', 'xcit', 'convnext', 'convnext_v2',
            'crossvit', 'tf_mobilenet', 'mobilevit', 'seresnet',
            'mnasnet', 'swin_v2', 'eva', 'eva_v2', 'tinynet', 'other',
            'resnet18_l2_adv', 'resnet50_l2_adv', 'wide_resnet50_2_l2_adv',
            'wide_resnet50_4_l2_adv', 'resnet18_linf_adv', 'resnet50_linf_adv', 'wide_resnet50_2_linf_adv', 'other_adv',
            'clip_1', 'clip_2'
        
        ]

    def get_num_models(self, name):
        return len(self.get_model_families(name))
    
    def check_timm_models_exists(self, name):
        model = timm.list_models(name, pretrained=True)
        return len(model)

    def get_model_families(self, name):
        if name == 'test':
            return ['resnet50.a3_in1k']
        
        if name == 'subset1':
            return ['resnet152.a3_in1k', 'resnet101.a3_in1k', 'resnet50.a3_in1k', 'resnet34.a3_in1k',  'resnet18.a3_in1k']
        
        if name == 'subset2':
            return ['vit_tiny_patch16_224.augreg_in21k_ft_in1k', 'vit_small_patch16_224.augreg_in1k', 'vit_small_patch16_224.augreg_in21k_ft_in1k', 
                    'vit_base_patch16_224.augreg_in21k_ft_in1k', 'vit_large_patch16_224.augreg_in21k_ft_in1k']
            
        if name == 'beit':
            return ['beit_base_patch16_224.in22k_ft_in22k_in1k', 'beitv2_base_patch16_224.in1k_ft_in1k', 'beitv2_base_patch16_224.in1k_ft_in22k_in1k', 
                    'beit_base_patch16_384.in22k_ft_in22k_in1k', 'beit_large_patch16_224.in22k_ft_in22k_in1k', 'beit_large_patch16_384.in22k_ft_in22k_in1k', 
                    'beit_large_patch16_512.in22k_ft_in22k_in1k', 'beitv2_large_patch16_224.in1k_ft_in1k', 'beitv2_large_patch16_224.in1k_ft_in22k_in1k']
   
        if name == 'caformer':
            return ['caformer_m36.sail_in1k', 'caformer_m36.sail_in22k_ft_in1k', 'caformer_s18.sail_in1k', 
                    'caformer_s18.sail_in22k_ft_in1k', 'caformer_s36.sail_in1k', 'caformer_s36.sail_in22k_ft_in1k']
            
        if name == 'cait':
            return ['cait_s24_224.fb_dist_in1k', 'cait_xxs24_224.fb_dist_in1k', 'cait_xxs36_224.fb_dist_in1k']
        
        if name == 'coat':
            return ['coat_lite_medium.in1k', 'coat_lite_mini.in1k', 'coat_lite_small.in1k', 'coat_lite_tiny.in1k', 
                    'coat_mini.in1k', 'coat_small.in1k', 'coat_tiny.in1k']
            
        if name == 'coatnet':
            return ['coatnet_0_rw_224.sw_in1k', 'coatnet_1_rw_224.sw_in1k', 'coatnet_bn_0_rw_224.sw_in1k', 
                    'coatnet_nano_rw_224.sw_in1k', 'coatnet_rmlp_1_rw_224.sw_in1k', 'coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k', 
                    'coatnet_rmlp_2_rw_224.sw_in1k', 'coatnet_rmlp_nano_rw_224.sw_in1k']
            
        if name == 'convit':
            return ['convit_base.fb_in1k', 'convit_small.fb_in1k', 'convit_tiny.fb_in1k'] 
        
        if name == 'convmixer':
            return ['convmixer_1024_20_ks9_p14.in1k', 'convmixer_1536_20.in1k', 'convmixer_768_32.in1k']
        
        if name == 'davit':
            return ['davit_base.msft_in1k', 'davit_small.msft_in1k', 'davit_tiny.msft_in1k']
        
        if name == 'deit':
            return ['deit_base_distilled_patch16_224.fb_in1k', 'deit_base_patch16_224.fb_in1k', 
                    'deit_small_distilled_patch16_224.fb_in1k', 'deit_small_patch16_224.fb_in1k', 
                    'deit_tiny_distilled_patch16_224.fb_in1k', 'deit_tiny_patch16_224.fb_in1k'] 
        
        if name == 'deit3':
            return ['deit3_base_patch16_224.fb_in1k', 'deit3_base_patch16_224.fb_in22k_ft_in1k', 'deit3_medium_patch16_224.fb_in1k',
                    'deit3_medium_patch16_224.fb_in22k_ft_in1k', 'deit3_small_patch16_224.fb_in1k', 'deit3_small_patch16_224.fb_in22k_ft_in1k']
                
        if name == 'densenet':       
            return ['densenet121.tv_in1k', 'densenet161.tv_in1k', 'densenet169.tv_in1k', 'densenet201.tv_in1k']
        
        if name == 'dla':
            return ['dla102.in1k', 'dla169.in1k', 'dla34.in1k', 'dla46_c.in1k', 'dla60.in1k']
        
        if name == 'dpn':
            return ['dpn107.mx_in1k', 'dpn131.mx_in1k', 'dpn68.mx_in1k', 'dpn68b.mx_in1k', 'dpn92.mx_in1k', 'dpn98.mx_in1k']
        
        if name == 'efficientformer':
            return ['efficientformer_l1.snap_dist_in1k', 'efficientformer_l3.snap_dist_in1k', 'efficientformer_l7.snap_dist_in1k', 
                    'efficientformerv2_l.snap_dist_in1k', 'efficientformerv2_s0.snap_dist_in1k', 'efficientformerv2_s1.snap_dist_in1k', 
                    'efficientformerv2_s2.snap_dist_in1k']
           
        if name == 'efficientnet':
            return ['efficientnet_b0.ra_in1k', 'efficientnet_es_pruned.in1k', 'efficientnet_es.ra_in1k', 'efficientnet_lite0.ra_in1k']
        
        if name == 'focalnet':
            return ['focalnet_base_srf.ms_in1k', 'focalnet_small_srf.ms_in1k', 'focalnet_tiny_srf.ms_in1k',
                    'focalnet_base_lrf.ms_in1k', 'focalnet_small_lrf.ms_in1k', 'focalnet_tiny_lrf.ms_in1k']
        
        if name == 'hrnet':
            return ['hrnet_w18_small_v2.ms_in1k', 'hrnet_w18_small.ms_in1k', 'hrnet_w18.ms_in1k', 'hrnet_w30.ms_in1k', 
                    'hrnet_w32.ms_in1k', 'hrnet_w40.ms_in1k', 'hrnet_w44.ms_in1k', 'hrnet_w48.ms_in1k']

        if name == 'lcnet':
            return ['lcnet_050.ra2_in1k', 'lcnet_075.ra2_in1k', 'lcnet_100.ra2_in1k']
        
        if name == 'levit':
            return ['levit_128.fb_dist_in1k', 'levit_128s.fb_dist_in1k', 'levit_192.fb_dist_in1k', 'levit_256.fb_dist_in1k', 'levit_384.fb_dist_in1k']
        
        if name == 'maxvit':
            return ['maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k', 'maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k', 'maxvit_rmlp_small_rw_224.sw_in1k', 'maxvit_small_tf_224.in1k', 'maxvit_tiny_rw_224.sw_in1k', 'maxvit_tiny_tf_224.in1k']
            
        if name == 'mixnet': 
            return ['mixnet_l.ft_in1k', 'mixnet_m.ft_in1k', 'mixnet_s.ft_in1k', 'mixnet_xl.ra_in1k']
        
        if name == 'mobilenet':   
            return ['mobilenetv2_050.lamb_in1k', 'mobilenetv2_100.ra_in1k', 'mobilenetv2_110d.ra_in1k', 'mobilenetv2_120d.ra_in1k', 
             'mobilenetv2_140.ra_in1k', 'mobilenetv3_large_100.miil_in21k_ft_in1k', 'mobilenetv3_large_100.ra_in1k', 
             'mobilenetv3_rw.rmsp_in1k', 'mobilenetv3_small_075.lamb_in1k', 'mobilenetv3_small_100.lamb_in1k']
            
        if name == 'mvit':
            return ['mvitv2_base.fb_in1k', 'mvitv2_small.fb_in1k', 'mvitv2_tiny.fb_in1k'] 
            
        if name == 'pit':
            return ['pit_b_224.in1k', 'pit_b_distilled_224.in1k', 'pit_s_224.in1k', 'pit_s_distilled_224.in1k', 'pit_ti_224.in1k', 
             'pit_ti_distilled_224.in1k', 'pit_xs_224.in1k', 'pit_xs_distilled_224.in1k']
            
        if name == 'poolformer':
            return ['poolformer_m36.sail_in1k', 'poolformer_s12.sail_in1k', 'poolformer_s24.sail_in1k', 'poolformer_s36.sail_in1k', 
             'poolformerv2_m36.sail_in1k', 'poolformerv2_s12.sail_in1k', 'poolformerv2_s24.sail_in1k', 'poolformerv2_s36.sail_in1k']
            
        if name == 'pvt':
            return ['pvt_v2_b0.in1k', 'pvt_v2_b1.in1k', 'pvt_v2_b2_li.in1k', 'pvt_v2_b2.in1k', 'pvt_v2_b3.in1k', 'pvt_v2_b4.in1k', 'pvt_v2_b5.in1k']
        
        if name == 'regnet':
            return ['regnetx_004_tv.tv2_in1k', 'regnetx_008.tv2_in1k', 'regnetx_016.tv2_in1k', 'regnety_004.tv2_in1k', 'regnety_008_tv.tv2_in1k', 'regnety_016.tv2_in1k']
        
        if name == 'repvgg':
            return ['repvgg_b0.rvgg_in1k','repvgg_b1.rvgg_in1k', 'repvgg_b1g4.rvgg_in1k', 'repvgg_b2.rvgg_in1k', 'repvgg_b2g4.rvgg_in1k', 'repvgg_b3g4.rvgg_in1k']
            
        if name == 'res2net':
            return ['res2net50_14w_8s.in1k', 'res2net50_26w_4s.in1k', 'res2net50_26w_6s.in1k', 'res2net50_26w_8s.in1k', 'res2net50_48w_2s.in1k']
            
        if name == 'resnest':
            return ['resnest50d_1s4x24d.in1k', 'resnest50d_4s2x40d.in1k', 'resnest50d.in1k']
            
        if name == 'resnet':
            return ['resnet101.a3_in1k', 'resnet10t.c3_in1k', 'resnet14t.c3_in1k', 'resnet152.a3_in1k', 'resnet18.a3_in1k', 
                    'resnet18.fb_ssl_yfcc100m_ft_in1k', 'resnet18.fb_swsl_ig1b_ft_in1k', 'resnet18.tv_in1k', 'resnet34.a3_in1k', 
                    'resnet50.a1h_in1k', 'resnet50.a3_in1k', 'resnet50.am_in1k', 'resnet50.tv_in1k', 'resnet50d.a3_in1k', 'resnetrs50.tf_in1k',
                    'resnet18.gluon_in1k', 'resnet34.gluon_in1k']
            
        if name == 'rexnet':
            return ['rexnet_100.nav_in1k', 'rexnet_130.nav_in1k', 'rexnet_150.nav_in1k', 'rexnet_200.nav_in1k', 'rexnet_300.nav_in1k']
        
        if name == 'sknet': 
            return ['skresnet18.ra_in1k', 'skresnet34.ra_in1k']
        
        if name == 'swin':
            return ['swin_base_patch4_window7_224.ms_in1k', 'swin_base_patch4_window7_224.ms_in22k_ft_in1k', 
                    'swin_s3_base_224.ms_in1k', 'swin_s3_small_224.ms_in1k', 'swin_s3_tiny_224.ms_in1k', 
                    'swin_small_patch4_window7_224.ms_in1k', 'swin_small_patch4_window7_224.ms_in22k_ft_in1k', 
                    'swin_tiny_patch4_window7_224.ms_in1k', 'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k',]
            
        if name == 'tresnet':
            return ['tresnet_l.miil_in1k', 'tresnet_m.miil_in1k', 'tresnet_m.miil_in21k_ft_in1k', 'tresnet_v2_l.miil_in21k_ft_in1k', 'tresnet_xl.miil_in1k']
        
        if name == 'twins':
            return ['twins_pcpvt_base.in1k', 'twins_pcpvt_large.in1k', 'twins_pcpvt_small.in1k', 'twins_svt_base.in1k', 'twins_svt_small.in1k']
        
        if name == 'vgg':
            return ['vgg11_bn.tv_in1k', 'vgg11.tv_in1k', 'vgg13_bn.tv_in1k', 'vgg13.tv_in1k', 'vgg16_bn.tv_in1k', 
                    'vgg16.tv_in1k', 'vgg19_bn.tv_in1k', 'vgg19.tv_in1k']
            
        if name == 'visformer':
            return ['visformer_small.in1k', 'visformer_tiny.in1k'] 
        
        if name == 'vit_base':
            return ['vit_base_patch16_224_miil.in21k_ft_in1k', 'vit_base_patch16_224.augreg_in1k', 'vit_base_patch16_224.augreg_in21k_ft_in1k', 
                    'vit_base_patch16_224.augreg2_in21k_ft_in1k', 'vit_base_patch16_224.orig_in21k_ft_in1k', 'vit_base_patch16_224.sam_in1k', 
                    'vit_base_patch16_rpn_224.sw_in1k', 'vit_base_patch8_224.augreg_in21k_ft_in1k', 'vit_base_patch8_224.augreg2_in21k_ft_in1k',]
            
        if name == 'vit_relpos':
            return [ 'vit_relpos_base_patch16_224.sw_in1k', 'vit_relpos_base_patch16_clsgap_224.sw_in1k', 'vit_relpos_medium_patch16_224.sw_in1k', 
                    'vit_relpos_medium_patch16_cls_224.sw_in1k', 'vit_relpos_medium_patch16_rpn_224.sw_in1k', 'vit_relpos_small_patch16_224.sw_in1k',]
                    
        if name == 'vit':
            return ['vit_small_patch16_224.augreg_in1k', 'vit_small_patch16_224.augreg_in21k_ft_in1k', 'vit_srelpos_medium_patch16_224.sw_in1k', 
                    'vit_srelpos_small_patch16_224.sw_in1k', 'vit_tiny_patch16_224.augreg_in21k_ft_in1k', 'vit_tiny_r_s16_p8_224.augreg_in21k_ft_in1k',]
        
        if name == 'vit_large':
            return ['vit_large_patch16_224.augreg_in21k_ft_in1k', 'vit_large_patch16_384.augreg_in21k_ft_in1k', 'vit_large_r50_s32_384.augreg_in21k_ft_in1k', 'vit_medium_patch16_gap_384.sw_in12k_ft_in1k'] 
            
        if name == 'volo':
            return ['volo_d1_384.sail_in1k', 'volo_d2_384.sail_in1k', 'volo_d3_224.sail_in1k', 'volo_d4_224.sail_in1k', 'volo_d5_224.sail_in1k', 'volo_d1_224.sail_in1k', 'volo_d2_224.sail_in1k'] 
            
        if name == 'xcit':
            return ['xcit_medium_24_p8_224.fb_in1k', 'xcit_nano_12_p8_224.fb_in1k', 'xcit_small_12_p8_224.fb_in1k', 'xcit_small_24_p8_224.fb_in1k', 'xcit_tiny_12_p8_224.fb_in1k', 'xcit_tiny_24_p8_224.fb_in1k']
            
        if name == 'convnext':
            return ['convnext_base.fb_in22k_ft_in1k' ,'convnext_base.fb_in22k_ft_in1k_384', 'convnext_large.fb_in22k_ft_in1k', 'convnext_large.fb_in22k_ft_in1k_384', 'convnext_small.in12k_ft_in1k_384']
        
        if name == 'convnext_v2':
            return ['convnextv2_base.fcmae_ft_in22k_in1k', 'convnextv2_base.fcmae_ft_in22k_in1k_384', 'convnextv2_large.fcmae_ft_in1k', 'convnextv2_large.fcmae_ft_in22k_in1k', 'convnextv2_large.fcmae_ft_in22k_in1k_384']
        
        if name == 'crossvit':
            return ['crossvit_9_240.in1k', 'crossvit_tiny_240.in1k']
        
        if name == 'tf_mobilenet':
            return ['tf_mobilenetv3_large_075.in1k', 'tf_mobilenetv3_large_minimal_100.in1k', 'tf_mobilenetv3_small_075.in1k', 'tf_mobilenetv3_small_100.in1k']
        
        if name == 'mobilevit':
            return ['mobilevit_xs.cvnets_in1k', 'mobilevit_xxs.cvnets_in1k', 'mobilevitv2_050.cvnets_in1k']
        
        if name == 'seresnet':
            return ['legacy_seresnet18.in1k', 'legacy_seresnet34.in1k', 'seresnextaa101d_32x8d.sw_in12k_ft_in1k', 'seresnextaa101d_32x8d.sw_in12k_ft_in1k_288']
        
        if name == 'mnasnet':
            return ['semnasnet_075.rmsp_in1k', 'mnasnet_100.rmsp_in1k', 'mnasnet_small.lamb_in1k']
        
        if name == 'swin_v2':
            return ['swinv2_base_window12to16_192to256.ms_in22k_ft_in1k', 'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k', 'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k', 'swinv2_large_window12to24_192to384.ms_in22k_ft_in1k',
            'swinv2_cr_small_224.sw_in1k', 'swinv2_cr_small_ns_224.sw_in1k', 'swinv2_cr_tiny_ns_224.sw_in1k']
            
        if name == 'eva':
            return ['eva_large_patch14_196.in22k_ft_in1k', 'eva_large_patch14_196.in22k_ft_in22k_in1k', 'eva_large_patch14_336.in22k_ft_in1k', 'eva_large_patch14_336.in22k_ft_in22k_in1k']
        
        if name == 'eva_v2':
            return ['eva02_base_patch14_448.mim_in22k_ft_in1k', 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', 'eva02_large_patch14_448.mim_in22k_ft_in1k', 'eva02_large_patch14_448.mim_in22k_ft_in22k_in1k', 'eva02_large_patch14_448.mim_m38m_ft_in1k']
            
        if name == 'tinynet':
            return ['tinynet_b.in1k', 'tinynet_c.in1k']
        
        if name == 'other':
            return ['edgenext_xx_small.in1k', 'ghostnet_100.in1k', 'spnasnet_100.rmsp_in1k']   
 
        if name == 'resnet18_l2_adv':
            return ['resnet18_l2_eps0.01', 'resnet18_l2_eps0.03', 'resnet18_l2_eps0.05', 'resnet18_l2_eps0.1', 'resnet18_l2_eps0.25',
                    'resnet18_l2_eps0.5', 'resnet18_l2_eps1', 'resnet18_l2_eps3', 'resnet18_l2_eps5']
        
        if name == 'resnet50_l2_adv':
            return ['resnet50_l2_eps0.01', 'resnet50_l2_eps0.03', 'resnet50_l2_eps0.05', 'resnet50_l2_eps0.1', 'resnet50_l2_eps0.25', 
                    'resnet50_l2_eps0.5', 'resnet50_l2_eps1', 'resnet50_l2_eps3', 'resnet50_l2_eps5']
        
        if name == 'wide_resnet50_2_l2_adv':
            return ['wide_resnet50_2_l2_eps0.01', 'wide_resnet50_2_l2_eps0.03', 'wide_resnet50_2_l2_eps0.05', 'wide_resnet50_2_l2_eps0.1', 'wide_resnet50_2_l2_eps0.25', 
                    'wide_resnet50_2_l2_eps0.5', 'wide_resnet50_2_l2_eps1', 'wide_resnet50_2_l2_eps3', 'wide_resnet50_2_l2_eps5']
        
        if name == 'wide_resnet50_4_l2_adv':
            # return ['wide_resnet50_4_l2_eps0.01', 'wide_resnet50_4_l2_eps0.03', 'wide_resnet50_4_l2_eps0.05', 'wide_resnet50_4_l2_eps0.1', 
            #         'wide_resnet50_4_l2_eps0.25', 'wide_resnet50_4_l2_eps0.5', 'wide_resnet50_4_l2_eps1', 'wide_resnet50_4_l2_eps3', 'wide_resnet50_4_l2_eps5']
            return []
        
        if name == 'resnet18_linf_adv':
            return ['resnet18_linf_eps0.5', 'resnet18_linf_eps1.0', 'resnet18_linf_eps2.0', 'resnet18_linf_eps4.0' ,'resnet18_linf_eps8.0']
        
        if name == 'resnet50_linf_adv':
            return ['resnet50_linf_eps0.5', 'resnet50_linf_eps1.0', 'resnet50_linf_eps2.0', 'resnet50_linf_eps4.0' ,'resnet50_linf_eps8.0']
        
        if name == 'wide_resnet50_2_linf_adv':
            return ['wide_resnet50_2_linf_eps0.5', 'wide_resnet50_2_linf_eps1.0', 'wide_resnet50_2_linf_eps2.0', 'wide_resnet50_2_linf_eps4.0', 'wide_resnet50_2_linf_eps8.0']
            
        if name == 'other_adv':
            # return ['densenet_l2_eps3', 'mnasnet_l2_eps3', 'mobilenet_l2_eps3', 'resnext50_32x4d_l2_eps3', 'shufflenet_l2_eps3', 'vgg16_bn_l2_eps3']
            return []
        
        if name == 'clip_1':
            return ['vit_large_patch14_clip_224.openai_ft_in12k_in1k','vit_large_patch14_clip_224.laion2b_ft_in12k_in1k','vit_large_patch14_clip_224.openai_ft_in1k',
                    'vit_large_patch14_clip_224.laion2b_ft_in1k','vit_base_patch16_clip_224.laion2b_ft_in12k_in1k',]
                    
        if name == 'clip_2':
            return ['vit_base_patch16_clip_224.openai_ft_in12k_in1k',
                    'vit_base_patch16_clip_224.laion2b_ft_in1k','vit_base_patch16_clip_224.openai_ft_in1k','vit_base_patch32_clip_224.laion2b_ft_in12k_in1k',
                    'vit_base_patch32_clip_224.laion2b_ft_in1k','vit_base_patch32_clip_224.openai_ft_in1k']
            
        if name == 'dino_v1':
            return ['vit_base_patch16_224.dino','vit_base_patch8_224.dino','vit_small_patch16_224.dino','vit_small_patch8_224.dino', 
            'resnet50.dino',"xcit_small_12_p16.dino", "xcit_medium_24_p16.dino",]
            # return ["xcit_small_12_p8.dino", "xcit_medium_24_p8.dino"]

        if name == 'dino_v2':
            return ['vit_giant_patch14_dinov2.lvd142m','vit_large_patch14_dinov2.lvd142m','vit_base_patch14_dinov2.lvd142m', 'vit_small_patch14_dinov2.lvd142m']
        
        return None
    
    def load_model(self, model_name, models_dir='/media/data_cifs/pfeng2/Adversarial_Alignment/models/'):
        ckpt_path = os.path.join(models_dir, model_name + '.ckpt')
        assert os.path.exists(ckpt_path), 'The model does not exist.'
        print('[', ckpt_path, '] is found!')
        
        if model_name.startswith('resnet18'):
            from torchvision.models import resnet18
            model = resnet18(pretrained=False)
            
        if model_name.startswith('resnet50'):
            from torchvision.models import resnet50
            model = resnet50(pretrained=False)
            
        if model_name.startswith('wide_resnet50_2'):
            from torchvision.models import wide_resnet50_2
            model = wide_resnet50_2(pretrained=False)
            
        checkpoint = torch.load(ckpt_path)
        sd = {k[len('module.model.'):]:v for k,v in checkpoint['model'].items() if k[:len('module.model.')] == 'module.model.'}  # Consider only the model and not normalizers or attacker
        model.load_state_dict(sd)
        return model

    def load_dinov1(self, model_name):
        return DINOv1(model_name)

    def load_dinov2(self, model_name):
        return DINOv2(model_name)

class MultilabelDataset(Dataset):
    def __init__(self, file_paths, img_transform):
        super(Dataset).__init__()
        self.file_paths = file_paths 
        self.preprocess = img_transform   
        
    def __getitem__(self, index):
        data = torch.load(self.file_paths[index])
        img, olabel, mlabel = data['image'], data['original_label'], torch.cat((data['correct_multi_labels'], data['unclear_multi_labels']), dim=0)
        
        img = img.to(torch.float32) / 255.0 # unit8 -> float32
        img = self.preprocess(img)

        mlabel = mlabel.to(torch.int64)       # int32 -> int64
        size = mlabel.shape[0]
        if size < 10:
            padding = torch.full((10 - size,), -1, dtype=mlabel.dtype)
            mlabel = torch.cat((mlabel, padding))
        elif size > 10:
            mlabel = mlabel[:10]
        else:
            pass

        # mlabel = torch.squeeze(mlabel)        # [batch_size, 1] -> [batch_size]
        # print(mlabel.shape)
        
        return img, olabel, mlabel
                
    def __len__(self):
        return len(self.file_paths)

if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",
                        required=True,
                        default='test',
                        help="Please see model_collections.py")
    parser.add_argument("-c", "--cuda", required=True, type=int, default=0,
                        choices=[0,1,2,3,4,5,6,7], help="Enter a GPU device id from 0 to 7")
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda:' + str(args.cuda))

    # Models
    models = Models()
    model_names = models.get_model_families(args.model)
    assert model_names != None, "Please check your model names!!!"
    model_num = models.get_num_models(args.model)
    print("Evaluating ", str(model_num), ' ', args.model, ' ...')

    # Data
    output_file = '/media/data_cifs/pfeng2/Pseudo_ClickMe/Results/multi_label_results.csv'
    data_dir = "/media/data_cifs/pfeng2/Harmoization/datasets/imagenet_multi_label"
    ckpt_cache = '/media/data_cifs/pfeng2/timm_model_zoo/'
    file_paths = glob.glob(os.path.join(data_dir, '*.pth')) 

    for i, model_name in enumerate(model_names):
        # Load models
        try:
            if args.model.endswith('adv'):
                model = models.load_model(model_name=model_name).to(device)
            elif 'dino_v1' in args.model:
                model = models.load_dinov1(model_name=model_name).to(device)
            elif 'dino_v2' in args.model:
                model = models.load_dinov2(model_name=model_name).to(device)
            else:
                model = timm.create_model(model_name, pretrained=True, num_classes=1000).to(device)
        except:
            print("Failed to create ", model_name)
            continue

        # model = models.load_dinov2(model_name=model_name).to(device)
        # model = models.load_dinov1(model_name=model_name).to(device)
        
        model.eval()

        # Get input configs
        data_config = timm.data.resolve_model_data_config(model)
        img_transform = create_transform(**data_config)
        img_transform = transforms.Compose(
            [transforms.ToPILImage()] + img_transform.transforms)
        # print(img_transform)
        
        # Create dataset
        dataset = MultilabelDataset(file_paths, img_transform)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, pin_memory=True)
        start = time.time()
        cnt = 0

        num_correct_per_class, num_images_per_class = {}, {}
        for batch_id, (img, olabel, mlabel) in enumerate(dataloader):
            print("  batch id: %s | %s/%s | %s | CUDA: %s\r" % (batch_id, str(i+1), str(model_num), model_name, str(args.cuda)), end = "")

            img, olabel, mlabel = img.to(device, non_blocking=True), olabel.to(device, non_blocking=True), mlabel.to(device, non_blocking=True)

            # The label of the image in ImageNet
            cur_class = olabel.item()

            # If we haven't processed this class yet, set the counters to 0
            if cur_class not in num_correct_per_class:
                num_correct_per_class[cur_class] = 0
                num_images_per_class[cur_class] = 0

            num_images_per_class[cur_class] += 1

            # Get the predictions for this image
            with torch.no_grad():
                output = model(img)
                # print(output.shape)
                cur_pred = torch.argmax(output, axis=-1) # get the index of the max log-probability)
                # print('\n',cur_pred, olabel)

            # Check prediction
            if torch.any(mlabel == cur_pred.item()):
                num_correct_per_class[cur_class] += 1

        acc_avg = 0
        num_classes = 1000
        assert len(num_correct_per_class) == num_classes
        assert len(num_images_per_class) == num_classes
        for cid in range(num_classes):
            acc_avg += num_correct_per_class[cid] / num_images_per_class[cid]
        acc_avg /= num_classes
   
        end = time.time()
        print("") 

        record = [model_name, round(acc_avg, 4), int(end-start)]
        print(record)

        # Write info
        write_csv_all(record, output_file)
        
        print(model_name, "has been evaluated!\n--------------------------")