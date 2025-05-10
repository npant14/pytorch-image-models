import torch
import torch.nn as nn
import numpy as np
import scipy as sp

def pad_to_size(tensor, target_size):
    """
    Center pad a 4D tensor to match the target spatial size.

    Args:
        tensor (Tensor): Input tensor (B, C, H, W).
        target_size (tuple): Desired size (H, W).

    Returns:
        Tensor: Padded tensor.
    """
    h, w = tensor.shape[-2:]
    pad_h = target_size[0] - h
    pad_w = target_size[1] - w
    pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    return nn.functional.pad(tensor, pad)


def get_gabor_filter_bank(l_size, l_div, n_ori, aspect_ratio=0.3):
    """
    Generate Gabor filters for different orientations.

    Args:
        l_size (int): Gabor size (filter dimension).
        l_div (float): Normalization divisor.
        n_ori (int): Number of orientations.
        aspect_ratio (float): Ellipticity of the Gaussian envelope.

    Returns:
        torch.Tensor: Shape (n_ori, 3, l_size, l_size)
    """
    la = l_size * 2 / l_div
    sigma = la * 0.8
    gs = l_size
    half_gs = (gs - 1) / 2.0

    thetas = np.linspace(0, np.pi, n_ori, endpoint=False) + np.pi / 2
    thetas = thetas[np.newaxis, np.newaxis, :]

    yy, xx = sp.mgrid[-half_gs:half_gs+1, -half_gs:half_gs+1]
    xx, yy = xx[:, :, np.newaxis], yy[:, :, np.newaxis]

    x = xx * np.cos(thetas) - yy * np.sin(thetas)
    y = xx * np.sin(thetas) + yy * np.cos(thetas)

    filt = np.exp(-(x ** 2 + (aspect_ratio * y) ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * x / la)
    filt[np.sqrt(x ** 2 + y ** 2) > gs / 2.] = 0

    # Normalize each orientation filter
    for ori in range(n_ori):
        filt[:, :, ori] -= filt[:, :, ori].mean()
        norm = np.linalg.norm(filt[:, :, ori])
        if norm != 0:
            filt[:, :, ori] /= norm

    filt = filt.astype(np.float32).transpose(2, 0, 1)  # (n_ori, H, W)
    filt = np.repeat(filt[:, np.newaxis, :, :], 3, axis=1)  # (n_ori, 3, H, W)
    return torch.tensor(filt)


class S1(nn.Module):
    """
    First stage of the HMAX model: applies orientation-selective Gabor filters.

    Args:
        scales (list): List of spatial scales (filter sizes).
        n_ori (int): Number of orientations.
        padding (str or int): Padding type for convolution.
        trainable_filters (bool): If True, Gabor filters are learnable.
        divs (list): Divisors for normalization per scale.
    """
    def __init__(self, scales, n_ori, padding, trainable_filters, divs):
        super(S1, self).__init__()
        assert len(scales) == len(divs), "Each scale must have a corresponding divisor"
        self.scales = scales
        self.divs = divs
        self.n_ori = n_ori

        for scale, div in zip(scales, divs):
            gabor = get_gabor_filter_bank(scale, div, n_ori)
            conv = nn.Conv2d(3, n_ori, scale, padding=padding, bias=False)
            conv.weight = nn.Parameter(gabor, requires_grad=trainable_filters)
            setattr(self, f's_{scale}', conv)

            # Uniform filter for local normalization
            uniform_conv = nn.Conv2d(3, n_ori, scale, padding=padding, bias=False)
            nn.init.constant_(uniform_conv.weight, 1.0)
            for param in uniform_conv.parameters():
                param.requires_grad = False
            setattr(self, f's_uniform_{scale}', uniform_conv)

    def forward(self, x):
        ori_size = x.shape[-2:]
        s1_maps = []

        for scale in self.scales:
            gabor_conv = getattr(self, f's_{scale}')
            norm_conv = getattr(self, f's_uniform_{scale}')

            response = torch.abs(gabor_conv(x))
            norm = torch.sqrt(torch.abs(norm_conv(x ** 2)))
            norm[norm == 0] = 1  # Avoid divide-by-zero
            response = response / norm

            padded = pad_to_size(response, ori_size)
            s1_maps.append(padded)

        return torch.stack(s1_maps, dim=4)  # Shape: B x C x H x W x S


class C(nn.Module):
    """
    C layer: performs scale-wise and spatial max pooling.

    Args:
        sp_kernel_size (list or int): Spatial pooling kernel sizes.
        sp_stride_factor (float or None): Stride factor for max pooling.
        n_in_sbands (int): Number of input scale bands.
        num_scales_pooled (int): Number of scale bands to pool at once.
        scale_stride (int): Step between pooled groups.
        image_subsample_factor (int): Factor for resizing outputs.
    """
    def __init__(self,
                 sp_kernel_size=list(range(8, 38, 4)),
                 sp_stride_factor=None,
                 n_in_sbands=None,
                 num_scales_pooled=2,
                 scale_stride=2,
                 image_subsample_factor=1):
        super(C, self).__init__()

        self.num_scales_pooled = num_scales_pooled
        self.scale_stride = scale_stride
        self.n_out_sbands = (n_in_sbands - num_scales_pooled) // scale_stride + 1
        self.img_subsample = image_subsample_factor

        if isinstance(sp_kernel_size, int):
            self.sp_kernel_size = [sp_kernel_size] * self.n_out_sbands
        else:
            self.sp_kernel_size = sp_kernel_size

        if len(self.sp_kernel_size) != self.n_out_sbands:
            raise ValueError('Mismatch between spatial kernel sizes and output scale bands')

        self.sp_stride = [
            int(0.5 + k / self.sp_kernel_size[0]) if sp_stride_factor else 1
            for k in self.sp_kernel_size
        ]

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError('Expected input shape: B x C x H x W x S')

        # Group input into overlapping scale groups
        grouped = [x[..., i::self.scale_stride][..., :self.n_out_sbands]
                   for i in range(self.num_scales_pooled)]
        x_grouped = torch.stack(grouped, dim=5)  # B x C x H x W x S x G
        x_max, _ = torch.max(x_grouped, dim=5)   # Max over scales

        # Max pooling over spatial dimensions
        pooled_maps = []
        ori_size = x_max.shape[2:4]

        for i, (k, s) in enumerate(zip(self.sp_kernel_size, self.sp_stride)):
            x_slice = x_max[..., i]

            if k >= 0:
                s = int(0.5 + self.sp_kernel_size[-1] / (k - k % 2))
                pooled = nn.functional.max_pool2d(x_slice, kernel_size=k, stride=s)
                pooled = pad_to_size(pooled, (ori_size[0] // self.img_subsample, ori_size[1] // self.img_subsample))
            else:
                # Global max pooling
                pooled = nn.functional.max_pool2d(x_slice, x_slice.shape[-2:], stride=1)

            pooled_maps.append(pooled)

        return torch.stack(pooled_maps, dim=4)  # B x C x H x W x S
    
    
class S2(nn.Module):
    """
    Applies learned filters over multiple scale bands with optional multiple kernel sizes.

    Args:
        channels_in (int): Number of input channels.
        channels_out (int): Number of output channels.
        kernel_size (int or list): Single kernel size or list of sizes.
        stride (int): Stride for convolution.
    """
    def __init__(self, channels_in, channels_out, kernel_size, stride):
        super(S2, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]

        self.kernel_size = sorted(kernel_size)
        for idx, k in enumerate(self.kernel_size):
            conv = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, k, stride),
                nn.BatchNorm2d(channels_out, eps=1e-3),
                nn.ReLU(True)
            )
            setattr(self, f's_{idx}', conv)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError("Expected 5D input: (B, C, H, W, S)")

        B, C, H, W, S = x.shape
        outputs = []

        for idx in range(len(self.kernel_size)):
            layer = getattr(self, f's_{idx}')
            band_outputs = []

            for i in range(S):
                s_map = layer(x[..., i])
                s_map = pad_to_size(s_map, (H, W))  # Match original spatial size
                band_outputs.append(s_map)

            outputs.append(torch.stack(band_outputs, dim=4))  # Keep scale dim

        return torch.cat(outputs, dim=1)  # Concatenate over channels


# Alias since S3 is the same as S2
S3 = S2

class HMAX_latest(nn.Module):
    """
    Hierarchical model inspired by HMAX with multi-scale, orientation-based processing.
    """
    def __init__(self,
                 s1_scales=range(7, 39, 2),
                 s1_divs=np.arange(4, 3.2, -0.05),
                 n_ori=4,
                 num_classes=1000,
                 s1_trainable_filters=False):
        super(HMAX_latest, self).__init__()

        self.s1_scales = list(s1_scales)
        self.s1_divs = list(s1_divs)
        self.n_ori = n_ori
        self.num_classes = num_classes

        # S1 → C1
        self.s1 = S1(self.s1_scales, n_ori, padding='valid', trainable_filters=s1_trainable_filters, divs=self.s1_divs)
        self.c1 = C(
            sp_kernel_size=self._get_kernel_sizes(self.s1_scales, 2, 2),
            sp_stride_factor=8,
            n_in_sbands=len(s1_scales),
            num_scales_pooled=2,
            scale_stride=2
        )

        # S2/C2 & S3/C3
        self.s2 = S2(n_ori, 64, kernel_size=3, stride=1)
        self.c2 = C(
            sp_kernel_size=self._get_kernel_sizes(self._get_kernel_sizes(self.s1_scales, 2, 2), 2, 2),
            n_in_sbands=8,
            num_scales_pooled=2,
            scale_stride=2
        )
        self.s3 = S3(64, 64, kernel_size=3, stride=1)
        self.c3 = C(
            sp_kernel_size=-1,  # Global pooling
            n_in_sbands=4,
            num_scales_pooled=4,
            scale_stride=2
        )

        # Bypass pathway: S2b + C2b
        self.s2b = S2(n_ori, 64, kernel_size=[6, 9, 12, 15], stride=1)
        self.c2b = C(
            sp_kernel_size=-1,
            n_in_sbands=8,
            num_scales_pooled=8,
            scale_stride=2
        )

        self.pool = nn.AdaptiveMaxPool2d(18)

        # S4 block
        self.s4 = nn.Sequential(
            nn.Conv2d(self._get_s4_input_channels(), 512, 1),
            nn.BatchNorm2d(512, eps=1e-3),
            nn.ReLU(True),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256, eps=1e-3),
            nn.ReLU(True),
            nn.MaxPool2d(2, 1)
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 256),
            nn.Linear(256, num_classes)
        )

    def _get_kernel_sizes(self, scales, num_pooled, stride):
        return [
            int(np.mean(scales[i:i + num_pooled]))
            for i in range(0, len(scales) - num_pooled + 1, stride)
        ]

    def _get_s4_input_channels(self):
        c1_ch = len(self.c1.sp_kernel_size) * self.n_ori
        c2_ch = len(self.c2.sp_kernel_size) * 64
        c3_ch = len(self.c3.sp_kernel_size) * 64
        c2b_ch = len(self.s2b.kernel_size) * len(self.c2b.sp_kernel_size) * 64
        return c1_ch + c2_ch + c3_ch + c2b_ch

    def _prepare_block(self, x):
        x = x.permute(0, 1, 4, 2, 3).reshape(x.size(0), -1, x.size(2), x.size(3))
        return self.pool(x)

    def forward(self, x):
        s1_maps = self.s1(x)
        c1_maps = self.c1(s1_maps)

        s2_maps = self.s2(c1_maps)
        c2_maps = self.c2(s2_maps)

        s3_maps = self.s3(c2_maps)
        c3_maps = self.c3(s3_maps)

        s2b_maps = self.s2b(c1_maps)
        c2b_maps = self.c2b(s2b_maps)

        c1_out = self._prepare_block(c1_maps)
        c2_out = self._prepare_block(c2_maps)
        c3_out = self._prepare_block(c3_maps)
        c2b_out = self._prepare_block(c2b_maps)

        s4_input = torch.cat([c1_out, c2_out, c3_out, c2b_out], dim=1)
        s4_out = self.s4(s4_input)

        return self.classifier(torch.flatten(s4_out, 1))


############################################################

# def visualize_map(map):
#     map = map.detach().numpy()
#     plt.imshow(map)


# def get_gabor(l_size, l_div, n_ori, aspect_ratio):
#     """generate the gabor filters
#     Args
#     ----
#         l_size: float
#             gabor sizes
#         l_div: floats
#             normalization value to be used
#         n_ori: type integer
#             number of orientations
#         aspect_ratio: type float
#             gabor aspect ratio
#     Returns
#     -------
#         gabor: type nparray
#             gabor filter
#     Example
#     -------
#         aspect_ratio  = 0.3
#         l_gabor_size = 7
#         l_div        = 4.0
#         n_ori         = 4
#         get_gabor(l_gabor_size, l_div, n_ori, aspect_ratio)
#     """

#     la = l_size * 2 / l_div
#     si = la * 0.8
#     gs = l_size

#     # TODO: inverse the axes in the begining so I don't need to do swap them back
#     # thetas for all gabor orientations
#     th = np.array(range(n_ori)) * np.pi / n_ori + np.pi / 2.
#     th = th[sp.newaxis, sp.newaxis, :]
#     hgs = (gs - 1) / 2.
#     yy, xx = sp.mgrid[-hgs: hgs + 1, -hgs: hgs + 1]
#     xx = xx[:, :, sp.newaxis]
#     yy = yy[:, :, sp.newaxis]

#     x = xx * np.cos(th) - yy * np.sin(th)
#     y = xx * np.sin(th) + yy * np.cos(th)

#     filt = np.exp(-(x ** 2 + (aspect_ratio * y) ** 2) / (2 * si ** 2)) * np.cos(2 * np.pi * x / la)
#     filt[np.sqrt(x ** 2 + y ** 2) > gs / 2.] = 0

#     # gabor normalization (following cns hmaxgray package)
#     for ori in range(n_ori):
#         filt[:, :, ori] -= filt[:, :, ori].mean()
#         filt_norm = fastnorm(filt[:, :, ori])
#         if filt_norm != 0: filt[:, :, ori] /= filt_norm
#     filt_c = np.array(filt, dtype='float32').swapaxes(0, 2).swapaxes(1, 2)

#     filt_c = torch.Tensor(filt_c)
#     filt_c = filt_c.view(n_ori, 1, gs, gs)
#     filt_c = filt_c.repeat((1, 3, 1, 1))

#     return filt_c


# def fastnorm(in_arr):
#     arr_norm = np.dot(in_arr.ravel(), in_arr.ravel()).sum() ** (1. / 2.)

#     return arr_norm


# def get_sp_kernel_sizes_C(scales, num_scales_pooled, scale_stride):
#     '''
#     Recursive function to find the right relative kernel sizes for the spatial pooling performed in a C layer.
#     The right relative kernel size is the average of the scales that will be pooled. E.g, if scale 7 and 9 will be
#     pooled, the kernel size for the spatial pool is 8 x 8
#     Parameters
#     ----------
#     scales
#     num_scales_pooled
#     scale_stride
#     Returns
#     -------
#     list of sp_kernel_size
#     '''

#     if len(scales) < num_scales_pooled:
#         return []
#     else:
#         average = int(sum(scales[0:num_scales_pooled]) / len(scales[0:num_scales_pooled]))
#         return [average] + get_sp_kernel_sizes_C(scales[scale_stride::], num_scales_pooled, scale_stride)



# class S1(nn.Module):
#     # TODO: make more flexible such that forward will accept any number of tensors
#     def __init__(self, scales, n_ori, padding, trainable_filters, divs):

#         super(S1, self).__init__()
#         assert (len(scales) == len(divs))
#         self.scales = scales
#         self.divs = divs

#         for scale, div in zip(self.scales, self.divs):
#             setattr(self, f's_{scale}', nn.Conv2d(3, n_ori, scale, padding=padding))
#             s1_cell = getattr(self, f's_{scale}')
#             gabor_filter = get_gabor(l_size=scale, l_div=div, n_ori=n_ori, aspect_ratio=0.3)
#             s1_cell.weight = nn.Parameter(gabor_filter, requires_grad=trainable_filters)

#             # For normalization
#             setattr(self, f's_uniform_{scale}', nn.Conv2d(3, n_ori, scale, bias=False))
#             s1_uniform = getattr(self, f's_uniform_{scale}')
#             nn.init.constant_(s1_uniform.weight, 1)
#             for param in s1_uniform.parameters():
#                 param.requires_grad = False

#     def forward(self, x):
#         s1_maps = []
#         # Loop over scales, normalizing.
#         for scale in self.scales:
#             s1_cell = getattr(self, f's_{scale}')
#             s1_map = torch.abs(s1_cell(x))  # adding absolute value

#             s1_unorm = getattr(self, f's_uniform_{scale}')
#             s1_unorm = torch.sqrt(abs(s1_unorm(x** 2)))
#             s1_unorm.data[s1_unorm == 0] = 1  # To avoid divide by zero
#             s1_map /= s1_unorm

#             s1_maps.append(s1_map)

#         # Padding (to get s1_maps in same size)
#         # TODO: figure out if we'll ever be in a scenario where we don't need the same padding left/right or top/down
#         # Or if the size difference is an odd number
#         ori_size = (x.shape[-2], x.shape[-1])
#         for i, s1_map in enumerate(s1_maps):
#             s1_maps[i] = pad_to_size(s1_map, ori_size)
#         s1_maps = torch.stack(s1_maps, dim=4)

#         return s1_maps


# class C(nn.Module):
#     # TODO: make more flexible such that forward will accept any number of tensors
#     def __init__(self, sp_kernel_size=list(range(8, 38, 4)), sp_stride_factor=None, n_in_sbands=None,
#                  num_scales_pooled=2, scale_stride=2,image_subsample_factor=1):

#         super(C, self).__init__()
#         self.sp_kernel_size = sp_kernel_size
#         self.num_scales_pooled = num_scales_pooled
#         self.sp_stride_factor = sp_stride_factor
#         self.scale_stride = scale_stride
#         self.n_in_sbands = n_in_sbands
#         self.n_out_sbands = int(((n_in_sbands - self.num_scales_pooled) / self.scale_stride) + 1)
#         self.img_subsample = image_subsample_factor 
#         # Checking
#         if type(self.sp_kernel_size) == int:
#             # Apply the same sp_kernel_size everywhere
#             self.sp_kernel_size = [self.sp_kernel_size] * self.n_out_sbands

#         if len(self.sp_kernel_size) != self.n_out_sbands:
#             raise ValueError('wrong number of sp_kernel_sizes provided')

#         # Compute strides
#         if self.sp_stride_factor is None:
#             self.sp_stride = [1] * self.n_out_sbands
#         else:
#             self.sp_stride = [int(0.5 + kernel_size/self.sp_kernel_size[0]) for kernel_size in self.sp_kernel_size]

#     def forward(self, x):
#         # TODO - make this whole section more memory efficient

#         # Evaluate input
#         if x.ndim != 5:
#             raise ValueError('expecting 5D input: BXCxHxWxS, where S is number of scalebands')

#         # Group scale bands to be pooled together
#         # TODO: deal with the scenario in which x cannot be split even
#         groups = []
#         for i in range(self.num_scales_pooled):
#             groups.append(x[:, :, :, :, i::self.scale_stride][:, :, :, :, 0:self.n_out_sbands])
#         x = torch.stack(groups, dim=5)

#         # Maxpool over scale groups
#         x, _ = torch.max(x, dim=5)

#         # Maxpool over positions
#         # TODO: deal with rectangular images if needed
#         c_maps = []
#         ori_size = x.shape[2:4]
#         for i, (kernel_size, stride) in enumerate(zip(self.sp_kernel_size, self.sp_stride)):
#             to_append = x[:, :, :, :, i]

#             if kernel_size >= 0:
#                 stride = int(0.5 + self.sp_kernel_size[-1]/(kernel_size-kernel_size%2)) 
#                 #print(kernel_size,stride)
#                 to_append = nn.functional.max_pool2d(to_append, kernel_size, stride)
#                 to_append = pad_to_size(to_append, (int(ori_size[0] /self.img_subsample), int(ori_size[1] / self.img_subsample)))
#             else:
#                 # Negative kernel_size indicating we want global
#                 to_append = nn.functional.max_pool2d(to_append, to_append.shape[-2], 1)
#             # TODO: think about whether the resolution gets too small here
#             # to_append = nn.functional.interpolate(to_append, orig_size)  # resizing to ensure all maps are same size
#             c_maps.append(to_append)

#         c_maps = torch.stack(c_maps, dim=4)

#         return c_maps


# class S2(nn.Module):
#     def __init__(self, channels_in, channels_out, kernel_size, stride):
#         super(S2, self).__init__()
#         if type(kernel_size) == int:
#             self.kernel_size = [kernel_size]
#             setattr(self, f's_0', nn.Sequential(nn.Conv2d(channels_in, channels_out, kernel_size, stride),
#                                                 nn.BatchNorm2d(channels_out, 1e-3),
#                                                 nn.ReLU(True)))
#         elif type(kernel_size) == list:
#             self.kernel_size = kernel_size
#             self.kernel_size.sort()
#             for i in range(len(kernel_size)):
#                 setattr(self, f's_{i}', nn.Sequential(nn.Conv2d(channels_in, channels_out, kernel_size[i], stride),
#                                                       nn.BatchNorm2d(channels_out, 1e-3),
#                                                       nn.ReLU(True)))

#     def forward(self, x):
#         # Evaluate input
#         if x.ndim != 5:
#             raise ValueError('expecting 5D input: BXCxHxWxS, where S is number of scalebands')

#         # Convolve each kernel with each scale band
#         # TODO: think of something more elegant than for loop
#         s_maps_per_k = []
#         ori_size = x.shape[2:4]
#         for k in range(len(self.kernel_size)):
#             s_maps_per_i = []
#             layer = getattr(self, f's_{k}')
#             for i in range(x.shape[-1]):  # assuming S is last dimension
#                 s_map = layer(x[:, :, :, :, i])
#                 # TODO: think about whether the resolution gets too small here
#                 s_map = pad_to_size(s_map, (ori_size[0], ori_size[1]))
#                 s_maps_per_i.append(s_map)
#             s_maps_per_k.append(torch.stack(s_maps_per_i, dim=4))
#         s_maps = torch.cat(s_maps_per_k, dim=1)
#         return s_maps


# class S3(S2):
#     # S3 does the same thing as S2
#     pass


# class HMAX_latest(nn.Module):
#     def __init__(self,
#                  s1_scales=range(7, 39, 2),
#                  s1_divs=np.arange(4, 3.2, -0.05),
#                  n_ori=4,
#                  num_classes=1000,
#                  s1_trainable_filters=False):
#         super(HMAX_latest, self).__init__()

#         # A few settings
#         self.s1_scales = s1_scales
#         self.s1_divs = s1_divs
#         self.n_ori = n_ori
#         self.num_classes = num_classes
#         self.s1_trainable_filters = s1_trainable_filters

#         self.c_scale_stride = 2
#         self.c_num_scales_pooled = 2

#         self.c1_scale_stride = self.c_scale_stride
#         self.c1_num_scales_pooled = self.c_num_scales_pooled
#         self.c1_sp_kernel_sizes = get_sp_kernel_sizes_C(self.s1_scales, self.c1_num_scales_pooled, self.c1_scale_stride)

#         self.c2_scale_stride = self.c_scale_stride
#         self.c2_num_scales_pooled = self.c_num_scales_pooled
#         self.c2_sp_kernel_sizes = get_sp_kernel_sizes_C(self.c1_sp_kernel_sizes, self.c2_num_scales_pooled, self.c2_scale_stride)

#         self.c2b_scale_stride = self.c_scale_stride
#         self.c2b_num_scales_pooled = len(self.c1_sp_kernel_sizes)  # all of them
#         # no kernel sizes here because global pool (spatially)

#         self.c3_scale_stride = self.c_scale_stride
#         self.c3_num_scales_pooled = len(self.c2_sp_kernel_sizes)  # all of them
#         # no kernel sizes here because global pool (spatially)

#         # Feature extractors (in the order of the table in Figure 1)
#         self.s1 = S1(scales=self.s1_scales, n_ori=n_ori, padding='valid', trainable_filters=s1_trainable_filters,
#                      divs=self.s1_divs)
#         self.c1 = C(sp_kernel_size=self.c1_sp_kernel_sizes, sp_stride_factor=8, n_in_sbands=len(s1_scales),
#                     num_scales_pooled=self.c1_num_scales_pooled, scale_stride=self.c1_scale_stride)
#         self.s2 = S2(channels_in=n_ori, channels_out=100, kernel_size=3, stride=1)
#         self.c2 = C(sp_kernel_size=self.c2_sp_kernel_sizes, sp_stride_factor=None, n_in_sbands=len(self.c1_sp_kernel_sizes),
#                     num_scales_pooled=self.c2_num_scales_pooled, scale_stride=self.c2_scale_stride)
#         self.s2b = S2(channels_in=n_ori, channels_out=100, kernel_size=[6, 9, 12, 15], stride=1)
#         self.s3 = S3(channels_in=100,channels_out=100, kernel_size=3, stride=1)
#         self.c2b = C(sp_kernel_size=-1, sp_stride_factor=None, n_in_sbands=len(self.c1_sp_kernel_sizes),
#                      num_scales_pooled=self.c2b_num_scales_pooled, scale_stride=self.c2b_scale_stride)
#         self.c3 = C(sp_kernel_size=-1, sp_stride_factor=None, n_in_sbands=len(self.c2_sp_kernel_sizes),
#                     num_scales_pooled=self.c3_num_scales_pooled, scale_stride=self.c3_scale_stride)
#         self.pool = nn.AdaptiveMaxPool2d(18)  # not in table, but we need to get everything in the same shape before s4

#         self.s4 = nn.Sequential(nn.Conv2d(self.get_s4_in_channels(), 512, 1, 1),
#                                 nn.BatchNorm2d(512, 1e-3),
#                                 nn.ReLU(True),
#                                 nn.Conv2d(512, 256, 1, 1),
#                                 nn.BatchNorm2d(256, 1e-3),
#                                 nn.ReLU(True),
#                                 nn.MaxPool2d(2, 1))

#         # Classifier
#         self.classifier = nn.Sequential(nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
#                                         #nn.Linear(256 * 8 * 8, 4096//8),  # fc1
#                                         #nn.BatchNorm1d(4096//8, 1e-3),
#                                         #nn.ReLU(True),
#                                         #nn.Dropout(0.3),  # TODO: check if this will be auto disabled if eval
#                                         nn.Linear(256 * 8 * 8, 256),  # fc2
#                                         #nn.BatchNorm1d(256, 1e-3),
#                                         #nn.ReLU(True),
#                                         nn.Linear(256, num_classes)  # fc3
#                                         )
        

#     def get_s4_in_channels(self):
#         c1_out = len(self.c1.sp_kernel_size) * self.n_ori
#         c2_out = len(self.c2.sp_kernel_size) * self.s2.s_0[0].weight.shape[0]
#         c3_out = len(self.c3.sp_kernel_size) * self.s3.s_0[0].weight.shape[0]
#         c2b_out = len(self.s2b.kernel_size) * len(self.c2b.sp_kernel_size) * self.s2b.s_0[0].weight.shape[0]
#         s4_in = c1_out + c2_out + c3_out + c2b_out
#         return s4_in

#     def forward(self, x):
#         s1_maps = self.s1(x)
#         c1_maps = self.c1(s1_maps)  # BxCxHxWxS with S number of scales

#         s2_maps = self.s2(c1_maps)
#         c2_maps = self.c2(s2_maps)
#         s3_maps = self.s3(c2_maps)
#         c3_maps = self.c3(s3_maps)

#         s2b_maps = self.s2b(c1_maps)
#         c2b_maps = self.c2b(s2b_maps)

#         # Prepare inputs for S4
#         c1_maps = torch.permute(c1_maps, (0, 1, 4, 2, 3))  # BxCxHxWxS --> BxCxSxHxW
#         c1_maps = c1_maps.reshape(c1_maps.shape[0], -1, *c1_maps.shape[3::])  # BxCxSxHxW --> Bx(C*S)xHxW
#         c1_maps = self.pool(c1_maps)

#         c2_maps = torch.permute(c2_maps, (0, 1, 4, 2, 3))  # BxCxHxWxS --> BxCxSxHxW
#         c2_maps = c2_maps.reshape(c2_maps.shape[0], -1, *c2_maps.shape[3::])  # BxCxSxHxW --> Bx(C*S)xHxW
#         c2_maps = self.pool(c2_maps)  # matching HxW across all inputs to S4

#         c3_maps = torch.permute(c3_maps, (0, 1, 4, 2, 3))  # BxCxHxWxS --> BxCxSxHxW
#         c3_maps = c3_maps.reshape(c3_maps.shape[0], -1, *c3_maps.shape[3::])  # BxCxSxHxW --> Bx(C*S)xHxW
#         c3_maps = self.pool(c3_maps)  # matching HxW across all inputs to S4

#         c2b_maps = torch.permute(c2b_maps, (0, 1, 4, 2, 3))  # BxCxHxWxS --> BxCxSxHxW
#         c2b_maps = c2b_maps.reshape(c2b_maps.shape[0], -1, *c2b_maps.shape[3::])  # BxCxSxHxW --> Bx(C*S)xHxW
#         c2b_maps = self.pool(c2b_maps)  # matching HxW across all inputs to S4

#         # Concatenate and pass to S4
#         s4_maps = self.s4(torch.cat([c1_maps, c2_maps, c3_maps, c2b_maps], dim=1))
#         s4_maps = torch.flatten(s4_maps, 1)

#         # Classify
#         output = self.classifier(s4_maps)

#         return output

# class HMAX_latest_slim(nn.Module):
#     def __init__(self,
#                     s1_scales=range(7, 39, 2),
#                     s1_divs=np.arange(4, 3.2, -0.05),
#                     n_ori=4,
#                     num_classes=1000,
#                     s1_trainable_filters=False):
#         super(HMAX_latest_slim, self).__init__()

#         # A few settings
#         self.s1_scales = s1_scales
#         self.s1_divs = s1_divs
#         self.n_ori = n_ori
#         self.num_classes = num_classes
#         self.s1_trainable_filters = s1_trainable_filters

#         self.c_scale_stride = 2
#         self.c_num_scales_pooled = 2

#         self.c1_scale_stride = self.c_scale_stride
#         self.c1_num_scales_pooled = self.c_num_scales_pooled
#         self.c1_sp_kernel_sizes = get_sp_kernel_sizes_C(self.s1_scales, self.c1_num_scales_pooled, self.c1_scale_stride)

#         self.c2_scale_stride = self.c_scale_stride
#         self.c2_num_scales_pooled = self.c_num_scales_pooled
#         self.c2_sp_kernel_sizes = get_sp_kernel_sizes_C(self.c1_sp_kernel_sizes, self.c2_num_scales_pooled, self.c2_scale_stride)

#         self.c2b_scale_stride = self.c_scale_stride
#         self.c2b_num_scales_pooled = len(self.c1_sp_kernel_sizes)  # all of them
#         # no kernel sizes here because global pool (spatially)

#         self.c3_scale_stride = self.c_scale_stride
#         self.c3_num_scales_pooled = len(self.c2_sp_kernel_sizes)  # all of them
#         # no kernel sizes here because global pool (spatially)

#         # Feature extractors (in the order of the table in Figure 1)
#         self.s1 = S1(scales=self.s1_scales, n_ori=n_ori, padding='valid', trainable_filters=s1_trainable_filters,
#                         divs=self.s1_divs)
#         self.c1 = C(sp_kernel_size=self.c1_sp_kernel_sizes, sp_stride_factor=8, n_in_sbands=len(s1_scales),
#                     num_scales_pooled=self.c1_num_scales_pooled, scale_stride=self.c1_scale_stride)
#         self.s2 = S2(channels_in=n_ori, channels_out=100, kernel_size=3, stride=1)
#         self.c2 = C(sp_kernel_size=self.c2_sp_kernel_sizes, sp_stride_factor=None, n_in_sbands=len(self.c1_sp_kernel_sizes),
#                     num_scales_pooled=self.c2_num_scales_pooled, scale_stride=self.c2_scale_stride)
#         self.s2b = S2(channels_in=n_ori, channels_out=100, kernel_size=[6, 9, 12, 15], stride=1)
#         self.s3 = S3(channels_in=100, channels_out=100, kernel_size=3, stride=1)
#         self.c2b = C(sp_kernel_size=-1, sp_stride_factor=None, n_in_sbands=len(self.c1_sp_kernel_sizes),
#                         num_scales_pooled=self.c2b_num_scales_pooled, scale_stride=self.c2b_scale_stride)
#         self.c3 = C(sp_kernel_size=-1, sp_stride_factor=None, n_in_sbands=len(self.c2_sp_kernel_sizes),
#                     num_scales_pooled=self.c3_num_scales_pooled, scale_stride=self.c3_scale_stride)
#         self.pool = nn.AdaptiveMaxPool2d(18)  # not in table, but we need to get everything in the same shape before s4

#         self.s4 = nn.Sequential(nn.Conv2d(self.get_s4_in_channels(), 512, 1, 1),
#                                 nn.BatchNorm2d(512, 1e-3),
#                                 nn.ReLU(True),
#                                 nn.Conv2d(512, 256, 1, 1),
#                                 nn.BatchNorm2d(256, 1e-3),
#                                 nn.ReLU(True),
#                                 nn.MaxPool2d(3, 2))

#         # Classifier
#         self.classifier = nn.Sequential(nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
#                                         nn.Linear(256 * 8 * 8, 4096),  # fc1
#                                         nn.BatchNorm1d(4096, 1e-3),
#                                         nn.ReLU(True),
#                                         nn.Dropout(0.5),  # TODO: check if this will be auto disabled if eval
#                                         nn.Linear(4096, 4096),  # fc2
#                                         nn.BatchNorm1d(4096, 1e-3),
#                                         nn.ReLU(True),
#                                         nn.Linear(4096, num_classes)  # fc3
#                                         )

#     def get_s4_in_channels(self):
#         c1_out = len(self.c1.sp_kernel_size) * self.n_ori
#         c2_out = len(self.c2.sp_kernel_size) * self.s2.s_0[0].weight.shape[0]
#         c3_out = len(self.c3.sp_kernel_size) * self.s3.s_0[0].weight.shape[0]
#         c2b_out = len(self.s2b.kernel_size) * len(self.c2b.sp_kernel_size) * self.s2b.s_0[0].weight.shape[0]
#         s4_in = c1_out + c2_out + c3_out + c2b_out
#         return s4_in

#     def forward(self, x):
#         s1_maps = self.s1(x)
#         c1_maps = self.c1(s1_maps)  # BxCxHxWxS with S number of scales

#         s2_maps = self.s2(c1_maps)
#         c2_maps = self.c2(s2_maps)
#         s3_maps = self.s3(c2_maps)
#         c3_maps = self.c3(s3_maps)

#         s2b_maps = self.s2b(c1_maps)
#         c2b_maps = self.c2b(s2b_maps)

#         # Prepare inputs for S4
#         c1_maps = torch.permute(c1_maps, (0, 1, 4, 2, 3))  # BxCxHxWxS --> BxCxSxHxW
#         c1_maps = c1_maps.reshape(c1_maps.shape[0], -1, *c1_maps.shape[3::])  # BxCxSxHxW --> Bx(C*S)xHxW
#         c1_maps = self.pool(c1_maps)

#         c2_maps = torch.permute(c2_maps, (0, 1, 4, 2, 3))  # BxCxHxWxS --> BxCxSxHxW
#         c2_maps = c2_maps.reshape(c2_maps.shape[0], -1, *c2_maps.shape[3::])  # BxCxSxHxW --> Bx(C*S)xHxW
#         c2_maps = self.pool(c2_maps)  # matching HxW across all inputs to S4

#         c3_maps = torch.permute(c3_maps, (0, 1, 4, 2, 3))  # BxCxHxWxS --> BxCxSxHxW
#         c3_maps = c3_maps.reshape(c3_maps.shape[0], -1, *c3_maps.shape[3::])  # BxCxSxHxW --> Bx(C*S)xHxW
#         c3_maps = self.pool(c3_maps)  # matching HxW across all inputs to S4

#         c2b_maps = torch.permute(c2b_maps, (0, 1, 4, 2, 3))  # BxCxHxWxS --> BxCxSxHxW
#         c2b_maps = c2b_maps.reshape(c2b_maps.shape[0], -1, *c2b_maps.shape[3::])  # BxCxSxHxW --> Bx(C*S)xHxW
#         c2b_maps = self.pool(c2b_maps)  # matching HxW across all inputs to S4

#         # Concatenate and pass to S4
#         s4_maps = self.s4(torch.cat([c1_maps, c2_maps, c3_maps, c2b_maps], dim=1))
#         s4_maps = torch.flatten(s4_maps, 1)

#         # Classify
#         output = self.classifier(s4_maps)

#         return output