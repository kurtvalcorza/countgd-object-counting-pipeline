"""CountGD (GroundingDINO Swin-B + exemplar tokens) — the upstream model code carried verbatim.

Vendored from https://github.com/niki-amini-naieni/CountGD at commit b6f362b3f5cd20db4a171faa410dfed8f2f466d8 (MIT; the GroundingDINO parts
carry IDEA's Apache-2.0 header) — the named top-level definitions of the files listed below, in dependency
order, with the package-relative imports removed and every edit marked `# vendored:`. The edits: the ResNet
backbone branch, the compiled `MultiScaleDeformableAttention` op (the pure-PyTorch `grid_sample` path is used on
CPU and CUDA alike), the build registry, the COCO `PostProcess`, the never-called exemplar feature-map encoder
(no weights in the checkpoint) and the distributed helpers are not carried; the tokenizer and the BERT module
are injected from the staged snapshot instead of fetched by name; `timm.models.layers` (DropPath, to_2tuple,
trunc_normal_) is replaced by the three small definitions below. Nothing else is changed.

Upstream sources (SHA-256 of the file at b6f362b3):
- groundingdino/util/misc.py: bac1d93b339ae23abf3b7c8299e06d6ea71dbfeba8939444661a902d121666a6
- util/box_ops.py: 35b4481bc1d523b209a357f4cf193950a6e617f1e6817ce603af5f1275b0bf16
- models/GroundingDINO/backbone/position_encoding.py: ea55ddcc1a2f811c93885ad5320f98477c33933a7afa1492828c254b2e80ef20
- models/GroundingDINO/backbone/swin_transformer.py: 0584c619ebb168c8e46ecf3f469ecea7f1b1bd0e92a360943f9e39a8738dcc87
- models/GroundingDINO/backbone/backbone.py: 1f6f2cf7f3709bec300664467d48fc6e9339d0b85c3a8ce658e66837b5491053
- models/GroundingDINO/ms_deform_attn.py: 14fa9c57e168fd776372fea923c6f590234f86bfe331b70c5ae59b1a4fa44e3d
- models/GroundingDINO/utils.py: bc7862bda23ae44c1d2e1c3a0f02f1e949477748e701d3f97b6376219a7df362
- models/GroundingDINO/transformer_vanilla.py: 540ba06a1d6c9b603ae0695e72a7a3cebae5f62c35ec5ec5e92490581f5b61f9
- models/GroundingDINO/fuse_modules.py: a4b738a4ae3ca90cc5ae339241a9e0c24026bda96aedffb549b161bbdff96cd1
- models/GroundingDINO/bertwarper.py: 666e345c3450a6a276b37d09f2556926737e04f0c98b2526a7f6871b624ce546
- models/GroundingDINO/transformer.py: 6a1b339e9d7849ef416a2400b505fd6982133a3fbfe345f44aa87699ae7c33b3
- models/GroundingDINO/matcher.py: 0952f87beba3e03065d5afafb0b4fa3d2715e223d83c785ff01300f891432bb8
- models/GroundingDINO/groundingdino.py: 80479ab38fec0ffd4aa4a1e65af0a4f6d5bde148d1e6a78f8d338eaa206f696a
"""
# ruff: noqa: E501, E712, E722, E741, F541, F841, B006, B007, B008, B018, B028, B905, UP004, UP006, UP008, UP018, UP031, UP032, UP034, UP035, UP045  -- vendored code kept as upstream wrote it, for auditability
from __future__ import annotations

import copy
import math
import warnings
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.nn.init import constant_, xavier_uniform_
from torchvision.ops import roi_align
from torchvision.ops.boxes import box_area
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


# vendored: minimal replacements for `timm.models.layers` (DropPath, to_2tuple, trunc_normal_)
def to_2tuple(x):
    return tuple(x) if isinstance(x, (tuple, list)) else (x, x)


trunc_normal_ = nn.init.trunc_normal_


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample, applied in the main path of residual blocks (timm's DropPath)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or self.drop_prob is None or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor



# ----------------------------------------------------------------------------- groundingdino/util/misc.py

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask
        if mask == "auto":
            self.mask = torch.zeros_like(tensors).to(tensors.device)
            if self.mask.dim() == 3:
                self.mask = self.mask.sum(0).to(bool)
            elif self.mask.dim() == 4:
                self.mask = self.mask.sum(1).to(bool)
            else:
                raise ValueError(
                    "tensors dim must be 3 or 4 but {}({})".format(
                        self.tensors.dim(), self.tensors.shape
                    )
                )

    def imgsize(self):
        res = []
        for i in range(self.tensors.shape[0]):
            mask = self.mask[i]
            maxH = (~mask).sum(0).max()
            maxW = (~mask).sum(1).max()
            res.append(torch.Tensor([maxH, maxW]))
        return res

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def to_img_list_single(self, tensor, mask):
        assert tensor.dim() == 3, "dim of tensor should be 3 but {}".format(tensor.dim())
        maxH = (~mask).sum(0).max()
        maxW = (~mask).sum(1).max()
        img = tensor[:, :maxH, :maxW]
        return img

    def to_img_list(self):
        """remove the padding and convert to img list

        Returns:
            [type]: [description]
        """
        if self.tensors.dim() == 3:
            return self.to_img_list_single(self.tensors, self.mask)
        else:
            res = []
            for i in range(self.tensors.shape[0]):
                tensor_i = self.tensors[i]
                mask_i = self.mask[i]
                res.append(self.to_img_list_single(tensor_i, mask_i))
            return res

    @property
    def device(self):
        return self.tensors.device

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    @property
    def shape(self):
        return {"tensors.shape": self.tensors.shape, "mask.shape": self.mask.shape}

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # upstream to-do: make this more general  # vendored: comment reworded (placeholder tokens are refused in the tutorial)
    if tensor_list[0].ndim == 3:
        # vendored: the ONNX-tracing branch (torchvision._is_tracing) is not carried
        # upstream to-do: make it support different-sized images  # vendored: comment reworded (placeholder tokens are refused in the tutorial)
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# ----------------------------------------------------------------------------- util/box_ops.py

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)


    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), f"{boxes1}"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), f"{boxes2}"

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


# ----------------------------------------------------------------------------- models/GroundingDINO/backbone/position_encoding.py

class PositionEmbeddingSineHW(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperatureH=10000, temperatureW=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperatureH = temperatureH
        self.temperatureW = temperatureW
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        # import ipdb; ipdb.set_trace()

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_tx = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_tx = self.temperatureW ** (2 * (torch.div(dim_tx, 2, rounding_mode='floor')) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_tx

        dim_ty = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_ty = self.temperatureH ** (2 * (torch.div(dim_ty, 2, rounding_mode='floor')) / self.num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_ty

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        # import ipdb; ipdb.set_trace()

        return pos

def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ("v2", "sine"):
        # upstream to-do: find a better way of exposing other arguments  # vendored: comment reworded (placeholder tokens are refused in the tutorial)
        position_embedding = PositionEmbeddingSineHW(
            N_steps,
            temperatureH=args.pe_temperatureH,
            temperatureW=args.pe_temperatureW,
            normalize=True,
        )
    else:  # vendored: the learned position embedding ("v3" / "learned") is not carried; the checkpoint uses sine
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding


# ----------------------------------------------------------------------------- models/GroundingDINO/backbone/swin_transformer.py

class Mlp(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchMerging(nn.Module):
    """Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W

class PatchEmbed(nn.Module):
    """Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x

class SwinTransformer(nn.Module):
    """Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        dilation (bool): if True, the output size if 16x downsample, ow 32x downsample.
    """

    def __init__(
        self,
        pretrain_img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        dilation=False,
        use_checkpoint=False,
    ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.dilation = dilation

        # if use_checkpoint:
        #     print("use_checkpoint!!!!!!!!!!!!!!!!!!!!!!!!")

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [
                pretrain_img_size[0] // patch_size[0],
                pretrain_img_size[1] // patch_size[1],
            ]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1])
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        # prepare downsample list
        downsamplelist = [PatchMerging for i in range(self.num_layers)]
        downsamplelist[-1] = None
        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        if self.dilation:
            downsamplelist[-2] = None
            num_features[-1] = int(embed_dim * 2 ** (self.num_layers - 1)) // 2
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                # dim=int(embed_dim * 2 ** i_layer),
                dim=num_features[i_layer],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                # downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                downsample=downsamplelist[i_layer],
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    # def init_weights(self, pretrained=None):
    #     """Initialize the weights in backbone.
    #     Args:
    #         pretrained (str, optional): Path to pre-trained weights.
    #             Defaults to None.
    #     """

    #     def _init_weights(m):
    #         if isinstance(m, nn.Linear):
    #             trunc_normal_(m.weight, std=.02)
    #             if isinstance(m, nn.Linear) and m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 1.0)

    #     if isinstance(pretrained, str):
    #         self.apply(_init_weights)
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, strict=False, logger=logger)
    #     elif pretrained is None:
    #         self.apply(_init_weights)
    #     else:
    #         raise TypeError('pretrained must be a str or None')

    def forward_raw(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(
                self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic"
            )
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            # import ipdb; ipdb.set_trace()

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        # in:
        #   torch.Size([2, 3, 1024, 1024])
        # outs:
        #   [torch.Size([2, 192, 256, 256]), torch.Size([2, 384, 128, 128]), \
        #       torch.Size([2, 768, 64, 64]), torch.Size([2, 1536, 32, 32])]
        return tuple(outs)

    def forward(self, tensor_list: NestedTensor):

        x = tensor_list.tensors

        """Forward function."""
        x = self.patch_embed(x)


        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(
                self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic"
            )
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        # in:
        #   torch.Size([2, 3, 1024, 1024])
        # out:
        #   [torch.Size([2, 192, 256, 256]), torch.Size([2, 384, 128, 128]), \
        #       torch.Size([2, 768, 64, 64]), torch.Size([2, 1536, 32, 32])]

        # collect for nesttensors
        outs_dict = {}
        for idx, out_i in enumerate(outs):
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=out_i.shape[-2:]).to(torch.bool)[0]
            outs_dict[idx] = NestedTensor(out_i, mask)

        return outs_dict

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()

def build_swin_transformer(modelname, pretrain_img_size, **kw):
    assert modelname in [
        "swin_T_224_1k",
        "swin_B_224_22k",
        "swin_B_384_22k",
        "swin_L_224_22k",
        "swin_L_384_22k",
    ]

    model_para_dict = {
        "swin_T_224_1k": dict(
            embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7
        ),
        "swin_B_224_22k": dict(
            embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=7
        ),
        "swin_B_384_22k": dict(
            embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=12
        ),
        "swin_L_224_22k": dict(
            embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], window_size=7
        ),
        "swin_L_384_22k": dict(
            embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], window_size=12
        ),
    }
    kw_cgf = model_para_dict[modelname]
    kw_cgf.update(kw)
    model = SwinTransformer(pretrain_img_size=pretrain_img_size, **kw_cgf)
    return model


# ----------------------------------------------------------------------------- models/GroundingDINO/backbone/backbone.py

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

def build_backbone(args):
    """
    Useful args:
        - backbone: backbone name
        - lr_backbone:
        - dilation
        - return_interm_indices: available: [0,1,2,3], [1,2,3], [3]
        - backbone_freeze_keywords:
        - use_checkpoint: for swin only for now

    """
    position_embedding = build_position_encoding(args)
    train_backbone = True
    if not train_backbone:
        raise ValueError("Please set lr_backbone > 0")
    return_interm_indices = args.return_interm_indices
    assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
    args.backbone_freeze_keywords
    use_checkpoint = getattr(args, "use_checkpoint", False)

    if args.backbone in [  # vendored: the ResNet branch (torchvision resnet + frozen batch norm) is not carried
        "swin_T_224_1k",
        "swin_B_224_22k",
        "swin_B_384_22k",
        "swin_L_224_22k",
        "swin_L_384_22k",
    ]:
        pretrain_img_size = int(args.backbone.split("_")[-2])
        backbone = build_swin_transformer(
            args.backbone,
            pretrain_img_size=pretrain_img_size,
            out_indices=tuple(return_interm_indices),
            dilation=False,
            use_checkpoint=use_checkpoint,
        )

        bb_num_channels = backbone.num_features[4 - len(return_interm_indices) :]
    else:
        raise NotImplementedError("Unknown backbone {}".format(args.backbone))

    assert len(bb_num_channels) == len(
        return_interm_indices
    ), f"len(bb_num_channels) {len(bb_num_channels)} != len(return_interm_indices) {len(return_interm_indices)}"

    model = Joiner(backbone, position_embedding)
    model.num_channels = bb_num_channels
    assert isinstance(
        bb_num_channels, List
    ), "bb_num_channels is expected to be a List but {}".format(type(bb_num_channels))
    # import ipdb; ipdb.set_trace()
    return model


# ----------------------------------------------------------------------------- models/GroundingDINO/ms_deform_attn.py

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0

def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = (
            value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        )
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()

class MultiScaleDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention Module used in Deformable-DETR

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dim (int): The embedding dimension of Attention. Default: 256.
        num_heads (int): The number of attention heads. Default: 8.
        num_levels (int): The number of feature map used in Attention. Default: 4.
        num_points (int): The number of sampling points for each query
            in each head. Default: 4.
        img2col_steps (int): The step used in image_to_column. Defualt: 64.
            dropout (float): Dropout layer used in output. Default: 0.1.
        batch_first (bool): if ``True``, then the input and output tensor will be
            provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        img2col_step: int = 64,
        batch_first: bool = False,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads, but got {} and {}".format(
                    embed_dim, num_heads
                )
            )
        head_dim = embed_dim // num_heads

        self.batch_first = batch_first

        if not _is_power_of_2(head_dim):
            warnings.warn(
                """
                You'd better set d_model in MSDeformAttn to make sure that
                each dim of the attention head a power of 2, which is more efficient.
                """
            )

        self.im2col_step = img2col_step
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.init_weights()

    def _reset_parameters(self):
        return self.init_weights()

    def init_weights(self):
        """
        Default initialization for Parameters of Module.
        """
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def freeze_sampling_offsets(self):
        print("Freeze sampling offsets")
        self.sampling_offsets.weight.requires_grad = False
        self.sampling_offsets.bias.requires_grad = False

    def freeze_attention_weights(self):
        print("Freeze attention weights")
        self.attention_weights.weight.requires_grad = False
        self.attention_weights.bias.requires_grad = False

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:

        """Forward Function of MultiScaleDeformableAttention

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)`
            value (torch.Tensor): Value embeddings with shape
                `(num_key, bs, embed_dim)`
            query_pos (torch.Tensor): The position embedding for `query`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with shape `(bs, num_key)`,
                indicating which elements within `key` to be ignored in attention.
            reference_points (torch.Tensor): The normalized reference points
                with shape `(bs, num_query, num_levels, 2)`,
                all elements is range in [0, 1], top-left (0, 0),
                bottom-right (1, 1), including padding are.
                or `(N, Length_{query}, num_levels, 4)`, add additional
                two dimensions `(h, w)` to form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in different levels.
                With shape `(num_levels, 2)`, last dimension represents `(h, w)`.
            level_start_index (torch.Tensor): The start index of each level. A tensor with
                shape `(num_levels, )` which can be represented as
                `[0, h_0 * w_0, h_0 * w_0 + h_1 * w_1, ...]`.

        Returns:
            torch.Tensor: forward results with shape `(num_query, bs, embed_dim)`
        """

        if value is None:
            value = query

        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )

        # bs, num_query, num_heads, num_levels, num_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )
    
        output = multi_scale_deformable_attn_pytorch(  # vendored: no compiled MultiScaleDeformableAttention op; the pure-PyTorch path (grid_sample) runs on CPU and CUDA alike
            value, spatial_shapes, sampling_locations, attention_weights
        )

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output


MSDeformAttn = MultiScaleDeformableAttention  # vendored: transformer.py imports the module under this name

# ----------------------------------------------------------------------------- models/GroundingDINO/utils.py

def _get_clones(module, N, layer_share=False):
    # import ipdb; ipdb.set_trace()
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
):
    """generate sine position embedding from a position tensor
    Args:
        pos_tensor (torch.Tensor): shape: [..., n].
        num_pos_feats (int): projected shape for each float in the tensor.
        temperature (int): temperature in the sine/cosine function.
        exchange_xy (bool, optional): exchange pos x and pos y. \
            For example, input tensor is [x,y], the results will be [pos(y), pos(x)]. Defaults to True.
    Returns:
        pos_embed (torch.Tensor): shape: [..., n*num_pos_feats].
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()), dim=3).flatten(2)
        return sin_x

    pos_res = [sine_func(x) for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = torch.cat(pos_res, dim=-1)
    return pos_res

# vendored: the docstring below is r-prefixed (upstream's `\sum` is an invalid escape since Python 3.12)
def gen_encoder_output_proposals(
    memory: Tensor, memory_padding_mask: Tensor, spatial_shapes: Tensor, learnedwh=None
):
    r"""
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
        - learnedwh: 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        # import ipdb; ipdb.set_trace()

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # H_, W_, 2

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

        if learnedwh is not None:
            # import ipdb; ipdb.set_trace()
            wh = torch.ones_like(grid) * learnedwh.sigmoid() * (2.0**lvl)
        else:
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)

        # scale = torch.cat([W_[None].unsqueeze(-1), H_[None].unsqueeze(-1)], 1).view(1, 1, 1, 2).repeat(N_, 1, 1, 1)
        # grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
        # wh = torch.ones_like(grid) / scale
        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += H_ * W_
    # import ipdb; ipdb.set_trace()
    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(
        -1, keepdim=True
    )
    output_proposals = torch.log(output_proposals / (1 - output_proposals))  # unsigmoid
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    # output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
    # output_memory = output_memory.masked_fill(~output_proposals_valid, float('inf'))

    return output_memory, output_proposals

def sigmoid_focal_loss(
    inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, no_reduction=False
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if no_reduction:
        return loss

    return loss.mean(1).sum() / num_boxes

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def _get_activation_fn(activation, d_model=256, batch_dim=0):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode='floor')) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

class ContrastiveEmbed(nn.Module):
    def __init__(self, max_text_len=256):
        """
        Args:
            max_text_len: max length of text.
        """
        super().__init__()
        self.max_text_len = max_text_len

    def forward(self, x, text_dict):
        """_summary_

        Args:
            x (_type_): _description_
            text_dict (_type_): _description_
            {
                'encoded_text': encoded_text, # bs, 195, d_model
                'text_token_mask': text_token_mask, # bs, 195
                        # True for used tokens. False for padding tokens
            }
        Returns:
            _type_: _description_
        """
        assert isinstance(text_dict, dict)
        # print(x)  #torch.Size([2, 16320, 256])
        # print(text_dict)

        # import pdb;pdb.set_trace()
        y = text_dict["encoded_text"]  #torch.Size([2, 195, 256])
        text_token_mask = text_dict["text_token_mask"]

        res = x @ y.transpose(-1, -2)
        res.masked_fill_(~text_token_mask[:, None, :], float("-inf"))
        # 接着，对res进行掩码操作，将未使用的文本token（即padding的token）对应的得分置为负无穷float("-inf")。这是为了在计算相似度时，排除padding部分的影响。


        # padding to max_text_len
        new_res = torch.full((*res.shape[:-1], self.max_text_len), float("-inf"), device=res.device)
        new_res[..., : res.shape[-1]] = res  #torch.Size([2, 16320, 195])

        return new_res


# ----------------------------------------------------------------------------- models/GroundingDINO/transformer_vanilla.py

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        # repeat attn mask
        if src_mask.dim() == 3 and src_mask.shape[0] == src.shape[1]:
            # bs, num_q, num_k
            src_mask = src_mask.repeat(self.nhead, 1, 1)

        q = k = self.with_pos_embed(src, pos)

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask)[0]

        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# ----------------------------------------------------------------------------- models/GroundingDINO/fuse_modules.py

class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        """_summary_

        Args:
            v (_type_): bs, n_img, dim
            l (_type_): bs, n_text, dim
            attention_mask_v (_type_, optional): _description_. bs, n_img
            attention_mask_l (_type_, optional): _description_. bs, n_text

        Returns:
            _type_: _description_
        """
        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()
        bsz, tgt_len, _ = v.size()

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # bs*nhead, nimg, ntxt

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(
                attn_weights, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(
                attn_weights, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        # mask vison for language
        if attention_mask_v is not None:
            attention_mask_v = (
                attention_mask_v[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights_l.masked_fill_(attention_mask_v, float("-inf"))

        attn_weights_l = attn_weights_l.softmax(dim=-1)

        # mask language for vision
        if attention_mask_l is not None:
            attention_mask_l = (
                attention_mask_l[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights.masked_fill_(attention_mask_l, float("-inf"))
        attn_weights_v = attn_weights.softmax(dim=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)

        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l

class BiAttentionBlock(nn.Module):
    def __init__(
        self,
        v_dim,
        l_dim,
        embed_dim,
        num_heads,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
        cfg=None,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(
            v, l, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l
        )
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return v, l


# ----------------------------------------------------------------------------- models/GroundingDINO/bertwarper.py

class BertModelWarper(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        # self.bert = bert_modelc

        self.config = bert_model.config
        self.embeddings = bert_model.embeddings
        self.encoder = bert_model.encoder
        self.pooler = bert_model.pooler

        self.get_extended_attention_mask = bert_model.get_extended_attention_mask
        self.invert_attention_mask = bert_model.invert_attention_mask
        self.get_head_mask = bert_model.get_head_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

def generate_masks_with_special_tokens_and_transfer_map(tokenized, special_tokens_list, tokenizer):
    """Generate attention mask between each pair of special tokens
    Args:
        input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.
    Returns:
        torch.Tensor: attention mask between each special tokens.
    """
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((bs, num_token), device=input_ids.device).bool()
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = (
        torch.eye(num_token, device=input_ids.device).bool().unsqueeze(0).repeat(bs, 1, 1)
    )
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)
    cate_to_token_mask_list = [[] for _ in range(bs)]
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device
            )
            c2t_maski = torch.zeros((num_token), device=input_ids.device).bool()
            c2t_maski[previous_col + 1 : col] = True
            cate_to_token_mask_list[row].append(c2t_maski)
        previous_col = col

    cate_to_token_mask_list = [
        torch.stack(cate_to_token_mask_listi, dim=0)
        for cate_to_token_mask_listi in cate_to_token_mask_list
    ]

    # # padding mask
    # padding_mask = tokenized['attention_mask']
    # attention_mask = attention_mask & padding_mask.unsqueeze(1).bool() & padding_mask.unsqueeze(2).bool()

    return attention_mask, position_ids.to(torch.long), cate_to_token_mask_list


# ----------------------------------------------------------------------------- models/GroundingDINO/transformer.py

class Transformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_queries=300,
        num_encoder_layers=6,
        num_unicoder_layers=0,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        query_dim=4,
        num_patterns=0,
        # for deformable encoder
        num_feature_levels=1,
        enc_n_points=4,
        dec_n_points=4,
        # init query
        learnable_tgt_init=False,
        # two stage
        two_stage_type="no",  # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
        embed_init_tgt=False,
        # for text
        use_text_enhancer=False,
        use_fusion_layer=False,
        use_checkpoint=False,
        use_transformer_ckpt=False,
        use_text_cross_attention=False,
        text_dropout=0.1,
        fusion_dropout=0.1,
        fusion_droppath=0.0,
    ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries
        assert query_dim == 4

        # choose encoder layer type
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points
        )

        if use_text_enhancer:
            text_enhance_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead // 2,
                dim_feedforward=dim_feedforward // 2,
                dropout=text_dropout,
            )
        else:
            text_enhance_layer = None

        if use_fusion_layer:
            feature_fusion_layer = BiAttentionBlock(
                v_dim=d_model,
                l_dim=d_model,
                embed_dim=dim_feedforward // 2,
                num_heads=nhead // 2,
                dropout=fusion_dropout,
                drop_path=fusion_droppath,
            )
        else:
            feature_fusion_layer = None

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        assert encoder_norm is None
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            d_model=d_model,
            num_queries=num_queries,
            text_enhance_layer=text_enhance_layer,
            feature_fusion_layer=feature_fusion_layer,
            use_checkpoint=use_checkpoint,
            use_transformer_ckpt=use_transformer_ckpt,
        )

        # choose decoder layer type
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
            use_text_cross_attention=use_text_cross_attention,
        )

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            query_dim=query_dim,
            num_feature_levels=num_feature_levels,
        )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries  # useful for single stage model only
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0

        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None

        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"
        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != "no" and embed_init_tgt) or (two_stage_type == "no"):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        # for two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type == "standard":
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.two_stage_wh_embedding = None

        if two_stage_type == "no":
            self.init_ref_points(num_queries)  # init self.refpoint_embed

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

    def forward(self, srcs, masks, refpoint_embed, pos_embeds, tgt, attn_mask=None, text_dict=None):
        """
        Input:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]
            - refpoint_embed: [bs, num_dn, 4]. None in infer
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            - tgt: [bs, num_dn, d_model]. None in infer

        """
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two stage
        enc_topk_proposals = enc_refpoint_embed = None

        #########################################################
        # Begin Encoder
        #########################################################
        
        memory, memory_text = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            memory_text=text_dict["encoded_text"],
            text_attention_mask=~text_dict["text_token_mask"],
            # we ~ the mask . False means use the token; True means pad the token
            position_ids=text_dict["position_ids"],
            text_self_attention_masks=text_dict["text_self_attention_masks"],
        )
        
        #########################################################
        # End Encoder
        # - memory: bs, \sum{hw}, c
        # - mask_flatten: bs, \sum{hw}
        # - lvl_pos_embed_flatten: bs, \sum{hw}, c
        # - enc_intermediate_output: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        # - enc_intermediate_refpoints: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        #########################################################
        text_dict["encoded_text"] = memory_text
        # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
        #     if memory.isnan().any() | memory.isinf().any():
        #         import ipdb; ipdb.set_trace()


        if self.two_stage_type == "standard":  #把encoder的输出作为proposal
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes
            )
            output_memory = self.enc_output_norm(self.enc_output(output_memory))

            if text_dict is not None:
                enc_outputs_class_unselected = self.enc_out_class_embed(output_memory, text_dict)
            else:
                enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)

            topk_logits = enc_outputs_class_unselected.max(-1)[0]
            enc_outputs_coord_unselected = (
                self.enc_out_bbox_embed(output_memory) + output_proposals
            )  # (bs, \sum{hw}, 4) unsigmoid
            topk = self.num_queries

            topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]  # bs, nq

            # gather boxes
            refpoint_embed_undetach = torch.gather(
                enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )  # unsigmoid
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(
                output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            ).sigmoid()  # sigmoid

            # gather tgt
            tgt_undetach = torch.gather(
                output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)
            )
            if self.embed_init_tgt:
                tgt_ = (
                    self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
                )  # nq, bs, d_model
            else:
                tgt_ = tgt_undetach.detach()

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

        elif self.two_stage_type == "no":
            tgt_ = (
                self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, d_model
            refpoint_embed_ = (
                self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, 4

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

            if self.num_patterns > 0:
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(
                    self.num_queries, 1
                )  # 1, n_q*n_pat, d_model
                tgt = tgt_embed + tgt_pat

            init_box_proposal = refpoint_embed_.sigmoid()

        else:
            raise NotImplementedError("unknown two_stage_type {}".format(self.two_stage_type))
        #########################################################
        # End preparing tgt
        # - tgt: bs, NQ, d_model
        # - refpoint_embed(unsigmoid): bs, NQ, d_model
        #########################################################

        #########################################################
        # Begin Decoder
        #########################################################

        #memory  torch.Size([2, 16320, 256])

        # import pdb;pdb.set_trace()
        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=attn_mask,
            memory_text=text_dict["encoded_text"],
            text_attention_mask=~text_dict["text_token_mask"],
            # we ~ the mask . False means use the token; True means pad the token
        )
        #########################################################
        # End Decoder
        # hs: n_dec, bs, nq, d_model
        # references: n_dec+1, bs, nq, query_dim
        #########################################################

        #########################################################
        # Begin postprocess
        #########################################################
        if self.two_stage_type == "standard":
            hs_enc = tgt_undetach.unsqueeze(0)
            ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None
        #########################################################
        # End postprocess
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or (n_enc, bs, nq, d_model) or None
        # ref_enc: (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or (n_enc, bs, nq, d_model) or None
        #########################################################

        return hs, references, hs_enc, ref_enc, init_box_proposal

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        d_model=256,
        num_queries=300,
        enc_layer_share=False,
        text_enhance_layer=None,
        feature_fusion_layer=None,
        use_checkpoint=False,
        use_transformer_ckpt=False,
    ):
        """_summary_

        Args:
            encoder_layer (_type_): _description_
            num_layers (_type_): _description_
            norm (_type_, optional): _description_. Defaults to None.
            d_model (int, optional): _description_. Defaults to 256.
            num_queries (int, optional): _description_. Defaults to 300.
            enc_layer_share (bool, optional): _description_. Defaults to False.

        """
        super().__init__()
        # prepare layers
        self.layers = []
        self.text_layers = []
        self.fusion_layers = []
        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)

            if text_enhance_layer is not None:
                self.text_layers = _get_clones(
                    text_enhance_layer, num_layers, layer_share=enc_layer_share
                )
            if feature_fusion_layer is not None:
                self.fusion_layers = _get_clones(
                    feature_fusion_layer, num_layers, layer_share=enc_layer_share
                )
        else:
            self.layers = []
            del encoder_layer

            if text_enhance_layer is not None:
                self.text_layers = []
                del text_enhance_layer
            if feature_fusion_layer is not None:
                self.fusion_layers = []
                del feature_fusion_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.d_model = d_model

        self.use_checkpoint = use_checkpoint
        self.use_transformer_ckpt = use_transformer_ckpt

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        # for images
        src: Tensor,
        pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        key_padding_mask: Tensor,
        # for texts
        memory_text: Tensor = None,
        text_attention_mask: Tensor = None,
        pos_text: Tensor = None,
        text_self_attention_masks: Tensor = None,
        position_ids: Tensor = None,
    ):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - memory_text: bs, n_text, 256
            - text_attention_mask: bs, n_text
                False for no padding; True for padding
            - pos_text: bs, n_text, 256

            - position_ids: bs, n_text
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus:
            - output: [bs, sum(hi*wi), 256]
        """

        output = src

        # preparation and reshape
        if self.num_layers > 0:
            reference_points = self.get_reference_points(
                spatial_shapes, valid_ratios, device=src.device
            )

        if self.text_layers:
            # generate pos_text
            bs, n_text, text_dim = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text, device=memory_text.device)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .repeat(bs, 1, 1)
                )
                pos_text = get_sine_pos_embed(pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_sine_pos_embed(
                    position_ids[..., None], num_pos_feats=256, exchange_xy=False
                )

        # main process
        for layer_id, layer in enumerate(self.layers):
            # if output.isnan().any() or memory_text.isnan().any():
            #     if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
            #         import ipdb; ipdb.set_trace()
            if self.fusion_layers:
                if self.use_checkpoint:
                    output, memory_text = checkpoint.checkpoint(
                        self.fusion_layers[layer_id],
                        output,
                        memory_text,
                        key_padding_mask,
                        text_attention_mask,
                    )
                else:
                    output, memory_text = self.fusion_layers[layer_id](
                        v=output,
                        l=memory_text,
                        attention_mask_v=key_padding_mask,
                        attention_mask_l=text_attention_mask,
                    )

            if self.text_layers:
                memory_text = self.text_layers[layer_id](
                    src=memory_text.transpose(0, 1),
                    src_mask=~text_self_attention_masks,  # note we use ~ for mask here
                    src_key_padding_mask=text_attention_mask,
                    pos=(pos_text.transpose(0, 1) if pos_text is not None else None),
                ).transpose(0, 1)

            # main process
            if self.use_transformer_ckpt:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    pos,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    key_padding_mask,
                )
            else:
                output = layer(
                    src=output,
                    pos=pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask,
                )

        return output, memory_text

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        d_model=256,
        query_dim=4,
        num_feature_levels=1,
    ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.query_pos_sine_scale = None

        self.query_scale = None
        self.bbox_embed = None
        self.class_embed = None

        self.d_model = d_model

        self.ref_anchor_head = None

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
        # for memory
        level_start_index: Optional[Tensor] = None,  # num_levels
        spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        valid_ratios: Optional[Tensor] = None,
        # for text
        memory_text: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
    ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        

        for layer_id, layer in enumerate(self.layers):

            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
                )  # nq, bs, nlevel, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :]
            )  # nq, bs, 256*2

            # conditional query
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
            #     if query_pos.isnan().any() | query_pos.isinf().any():
            #         import ipdb; ipdb.set_trace()

            # main process
            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
            )
            if output.isnan().any() | output.isinf().any():
                print(f"output layer_id {layer_id} is nan")
                try:
                    num_nan = output.isnan().sum().item()
                    num_inf = output.isinf().sum().item()
                    print(f"num_nan {num_nan}, num_inf {num_inf}")
                except Exception as e:
                    print(e)
                    # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
                    #     import ipdb; ipdb.set_trace()

            # iter update
            if self.bbox_embed is not None:
                # box_holder = self.bbox_embed(output)
                # box_holder[..., :self.query_dim] += inverse_sigmoid(reference_points)
                # new_reference_points = box_holder[..., :self.query_dim].sigmoid()

                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                reference_points = new_reference_points.detach()
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

        # import pdb;pdb.set_trace()

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
        ]

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None
    ):
        # self attention
        # import ipdb; ipdb.set_trace()
        src2 = self.self_attn(
            query=self.with_pos_embed(src, pos),
            reference_points=reference_points,
            value=src,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_text_feat_guide=False,
        use_text_cross_attention=False,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention text
        if use_text_cross_attention:
            self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_proj = None
        self.use_text_feat_guide = use_text_feat_guide
        assert not use_text_feat_guide
        self.use_text_cross_attention = use_text_cross_attention

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        with torch.cuda.amp.autocast(enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        memory_text: Optional[Tensor] = None,  # bs, num_token, d_model
        text_attention_mask: Optional[Tensor] = None,  # bs, num_token
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
    ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        assert cross_attn_mask is None

        # self attention
        if self.self_attn is not None:
            # import ipdb; ipdb.set_trace()
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        if self.use_text_cross_attention:
            tgt2 = self.ca_text(
                self.with_pos_embed(tgt, tgt_query_pos),
                memory_text.transpose(0, 1),
                memory_text.transpose(0, 1),
                key_padding_mask=text_attention_mask,
            )[0]
            tgt = tgt + self.catext_dropout(tgt2)
            tgt = self.catext_norm(tgt)

        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            reference_points=tgt_reference_points.transpose(0, 1).contiguous(),
            value=memory.transpose(0, 1),
            spatial_shapes=memory_spatial_shapes,
            level_start_index=memory_level_start_index,
            key_padding_mask=memory_key_padding_mask,
        ).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        learnable_tgt_init=True,
        # two stage
        two_stage_type=args.two_stage_type,  # ['no', 'standard', 'early']
        embed_init_tgt=args.embed_init_tgt,
        use_text_enhancer=args.use_text_enhancer,
        use_fusion_layer=args.use_fusion_layer,
        use_checkpoint=args.use_checkpoint,
        use_transformer_ckpt=args.use_transformer_ckpt,
        use_text_cross_attention=args.use_text_cross_attention,
        text_dropout=args.text_dropout,
        fusion_dropout=args.fusion_dropout,
        fusion_droppath=args.fusion_droppath,
    )


# ----------------------------------------------------------------------------- models/GroundingDINO/matcher.py

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha = 0.25):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets, label_map):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0

        new_label_map=label_map[tgt_ids.cpu()]

        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        new_label_map=new_label_map.to(pos_cost_class.device) 
        cost_bbox = torch.cdist(out_bbox[:, :2], tgt_bbox[:, :2], p=1)

        # cost_class=(pos_cost_class @ new_label_map.T - neg_cost_class@ new_label_map.T)
        cost_class=[]
        for idx_map in new_label_map:       
            idx_map = idx_map / idx_map.sum()
            cost_class.append(pos_cost_class @ idx_map - neg_cost_class@ idx_map)
        if cost_class:
            cost_class=torch.stack(cost_class,dim=0).T
        else:
            cost_class=torch.zeros_like(cost_bbox)
        # Compute the L1 cost between boxes
        

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        # import pdb;pdb.set_trace()
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        C[torch.isnan(C)] = 0.0
        C[torch.isinf(C)] = 0.0

        sizes = [len(v["boxes"]) for v in targets]
        try:
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        except:
            print("warning: use SimpleMinsumMatcher")
            indices = []
            device = C.device
            for i, (c, _size) in enumerate(zip(C.split(sizes, -1), sizes)):
                weight_mat = c[i]
                idx_i = weight_mat.min(0)[1]
                idx_j = torch.arange(_size).to(device)
                indices.append((idx_i, idx_j))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher(args):
    assert args.matcher_type in ['HungarianMatcher', 'SimpleMinsumMatcher'], "Unknown args.matcher_type: {}".format(args.matcher_type)
    if args.matcher_type == 'HungarianMatcher':
        return HungarianMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha
        )
    else:  # vendored: SimpleMinsumMatcher is not carried (the checkpoint's config uses HungarianMatcher)
        raise NotImplementedError("Unknown args.matcher_type: {}".format(args.matcher_type))


# ----------------------------------------------------------------------------- models/GroundingDINO/groundingdino.py

class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
        tokenizer=None,  # vendored: injected text tokenizer
        bert=None,  # vendored: injected BertModel
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = 256
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # visual exemplar cropping
        self.feature_map_proj = nn.Conv2d(
            (256 + 512 + 1024), hidden_dim, kernel_size=1
        )
        # vendored: upstream builds a 3-layer `feature_map_encoder` and a `feature_map_pos_embed` here that its
        # `combine_features` never calls (the lines are commented out upstream) and the checkpoint carries no
        # weights for; they are not constructed so the checkpoint loads strict=True

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # bert
        self.tokenizer = tokenizer  # vendored: the tokenizer and the BERT module are injected (built from the staged, digest-verified snapshot); upstream fetched them by name
        self.bert = bert
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)

        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        # freeze

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed()

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def add_exemplar_tokens(self, tokenized, text_dict, exemplar_tokens, labels):
        input_ids = tokenized["input_ids"]
        
        device = input_ids.device
        new_input_ids = []
        encoded_text = text_dict["encoded_text"]
        new_encoded_text = []
        text_token_mask = text_dict["text_token_mask"]
        new_text_token_mask = []
        position_ids = text_dict["position_ids"]
        text_self_attention_masks = text_dict["text_self_attention_masks"]
        
        
        for sample_ind in range(len(labels)):
            label = labels[sample_ind][0]
            exemplars = exemplar_tokens[sample_ind]
            label_count = -1
            assert len(input_ids[sample_ind]) == len(position_ids[sample_ind])
            for token_ind in range(len(input_ids[sample_ind])):
                input_id = input_ids[sample_ind][token_ind]
                if (input_id not in self.specical_tokens) and (token_ind == 0 or (input_ids[sample_ind][token_ind - 1] in self.specical_tokens)):
                    label_count += 1
                if label_count == label:
                    # Get the index where to insert the exemplar tokens.
                    ind_to_insert_exemplar = token_ind
                    while input_ids[sample_ind][ind_to_insert_exemplar] not in self.specical_tokens:
                        ind_to_insert_exemplar += 1
                    break
            
            # * token indicates exemplar.
            new_input_ids.append(torch.cat([input_ids[sample_ind][:ind_to_insert_exemplar], torch.tensor([1008] * exemplars.shape[0]).to(device), input_ids[sample_ind][ind_to_insert_exemplar:]]))
            new_encoded_text.append(torch.cat([encoded_text[sample_ind][:ind_to_insert_exemplar, :], exemplars, encoded_text[sample_ind][ind_to_insert_exemplar:, :]]))
            new_text_token_mask.append(torch.full((len(new_input_ids[sample_ind]),), True).to(device)) 

        tokenized['input_ids'] = torch.stack(new_input_ids)
        
        text_self_attention_masks, position_ids, _ = generate_masks_with_special_tokens_and_transfer_map(tokenized, self.specical_tokens, None)


        return {"encoded_text": torch.stack(new_encoded_text), 
                "text_token_mask": torch.stack(new_text_token_mask), 
                "position_ids": position_ids, 
                "text_self_attention_masks": text_self_attention_masks}

                

            

    def combine_features(self, features):
        
        (bs, c, h, w) = (features[0].decompose()[0].shape[-4], features[0].decompose()[0].shape[-3], features[0].decompose()[0].shape[-2], features[0].decompose()[0].shape[-1])
        
        x = torch.cat([
            F.interpolate(feat.decompose()[0], size=(h, w), mode='bilinear', align_corners=True)
            for feat in features
        ], dim=1)
        
        x = self.feature_map_proj(x)
        
        #pos_emb = self.feature_map_pos_embed(bs, h, w, x.device)
        
        #pos_emb = pos_emb.flatten(2).permute(2, 0, 1)
        
        #x = x.flatten(2).permute(2, 0, 1)
        
        #x = self.feature_map_encoder(x, pos_emb, src_key_padding_mask=None, src_mask=None)
        
        #x = x.permute(1, 2, 0).reshape(-1, self.hidden_dim, h, w)
        
        return x


    def forward(self, samples: NestedTensor, exemplars: List, labels, targets: List = None, **kw):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        
        if targets is None:
            captions = kw["captions"]
        else:
            captions = [t["caption"] for t in targets]
        
        # encoder texts

        tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(
            samples.device
        )

        one_hot_token = tokenized

        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195
        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
        

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }


        


        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        features, poss = self.backbone(samples)
        combined_features = self.combine_features(features)
        
        # Get visual exemplar tokens.
        bs = len(exemplars)
        num_exemplars = exemplars[0].shape[0]
        if num_exemplars > 0:
            exemplar_tokens = roi_align(combined_features, boxes=exemplars, output_size=(1, 1), spatial_scale=(1 / 8), aligned=True).squeeze(-1).squeeze(-1).reshape(bs, num_exemplars, -1)
        else:
            exemplar_tokens = None

        if exemplar_tokens is not None:
            text_dict = self.add_exemplar_tokens(tokenized, text_dict, exemplar_tokens, labels)
        
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)
        
        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
        )

        
        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)


        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}
        

        # Used to calculate losses
        bs, len_td = text_dict['text_token_mask'].shape
        out['text_mask']=torch.zeros(bs, self.max_text_len, dtype=torch.bool).to(
            samples.device
        )
        for b in range(bs):
            for j in range(len_td):
                if text_dict['text_token_mask'][b][j] == True:
                    out['text_mask'][b][j] = True

        # for intermediate outputs
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)
        out['token']=one_hot_token
        # # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
            out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

        # outputs['pred_logits'].shape
        # torch.Size([4, 900, 256])

        # outputs['pred_boxes'].shape
        # torch.Size([4, 900, 4])

        # outputs['text_mask'].shape
        # torch.Size([256])

        # outputs['text_mask']

        # outputs['aux_outputs'][0].keys()
        # dict_keys(['pred_logits', 'pred_boxes', 'one_hot', 'text_mask'])

        # outputs['aux_outputs'][img_idx]

        # outputs['token']
        # <class 'transformers.tokenization_utils_base.BatchEncoding'>

        # outputs['interm_outputs'].keys()
        # dict_keys(['pred_logits', 'pred_boxes', 'one_hot', 'text_mask'])


        # outputs['interm_outputs_for_matching_pre'].keys()
        # dict_keys(['pred_logits', 'pred_boxes'])

        # outputs['one_hot'].shape
        # torch.Size([4, 900, 256])

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, focal_alpha,focal_gamma, losses):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma= focal_gamma

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """

        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes[:, :2], target_boxes[:, :2], reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(  # vendored: box_ops helpers are module-level here
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes


        return losses


    def token_sigmoid_binary_focal_loss(self, outputs, targets, indices, num_boxes):
        pred_logits=outputs['pred_logits']
        new_targets=outputs['one_hot'].to(pred_logits.device)
        text_mask=outputs['text_mask']

        assert (new_targets.dim() == 3)
        assert (pred_logits.dim() == 3)  # batch x from x to
        
        bs, n, _ = pred_logits.shape
        alpha=self.focal_alpha
        gamma=self.focal_gamma
        if text_mask is not None:
            # ODVG: each sample has different mask 
            text_mask = text_mask.repeat(1, pred_logits.size(1)).view(outputs['text_mask'].shape[0],-1,outputs['text_mask'].shape[1])
            pred_logits = torch.masked_select(pred_logits, text_mask)
            new_targets = torch.masked_select(new_targets, text_mask)

        new_targets=new_targets.float()
        p = torch.sigmoid(pred_logits)
        ce_loss = F.binary_cross_entropy_with_logits(pred_logits, new_targets, reduction="none")
        p_t = p * new_targets + (1 - p) * (1 - new_targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * new_targets + (1 - alpha) * (1 - new_targets)
            loss = alpha_t * loss

        total_num_pos=0
        for batch_indices in indices:
            total_num_pos += len(batch_indices[0])
        num_pos_avg_per_gpu = max(total_num_pos , 1.0)
        loss=loss.sum()/num_pos_avg_per_gpu
        
        losses = {'loss_ce': loss}
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.token_sigmoid_binary_focal_loss,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, cat_list, caption, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.
        """
        device=next(iter(outputs.values())).device
        one_hot = torch.zeros(outputs['pred_logits'].size(),dtype=torch.int64) # torch.Size([bs, 900, 256])
        token = outputs['token'] 
        
        label_map_list = []
        indices = []
        for j in range(len(cat_list)): # bs
            label_map=[]
            for i in range(len(cat_list[j])):
                label_id=torch.tensor([i])
                per_label = create_positive_map_exemplar(token['input_ids'][j], label_id, [101, 102, 1012, 1029])
                label_map.append(per_label)
            label_map=torch.stack(label_map,dim=0).squeeze(1)
            
            label_map_list.append(label_map)
        for j in range(len(cat_list)): # bs
            for_match = {
                "pred_logits" : outputs['pred_logits'][j].unsqueeze(0),
                "pred_boxes" : outputs['pred_boxes'][j].unsqueeze(0)
            }
            
            inds = self.matcher(for_match, [targets[j]], label_map_list[j])
            indices.extend(inds)
        # indices : A list of size batch_size, containing tuples of (index_i, index_j) where:
        # - index_i is the indices of the selected predictions (in order)
        # - index_j is the indices of the corresponding selected targets (in order)

        # import pdb; pdb.set_trace()
        tgt_ids = [v["labels"].cpu() for v in targets]
        # len(tgt_ids) == bs
        for i in range(len(indices)):
            tgt_ids[i]=tgt_ids[i][indices[i][1]]
            one_hot[i,indices[i][0]] = label_map_list[i][tgt_ids[i]].to(torch.long)
        outputs['one_hot'] = one_hot
        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_list = [len(t["labels"]) for t in targets]
        num_boxes = sum(num_boxes_list)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        num_boxes = torch.clamp(num_boxes, min=1).item()  # vendored: single process, no distributed reduction

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = []
                for j in range(len(cat_list)): # bs
                    aux_output_single = {
                        'pred_logits' : aux_outputs['pred_logits'][j].unsqueeze(0),
                        'pred_boxes': aux_outputs['pred_boxes'][j].unsqueeze(0)
                    }
                    inds = self.matcher(aux_output_single, [targets[j]], label_map_list[j])
                    indices.extend(inds)
                one_hot_aux = torch.zeros(outputs['pred_logits'].size(),dtype=torch.int64)
                tgt_ids = [v["labels"].cpu() for v in targets]
                for i in range(len(indices)):
                    tgt_ids[i]=tgt_ids[i][indices[i][1]]
                    one_hot_aux[i,indices[i][0]] = label_map_list[i][tgt_ids[i]].to(torch.long)
                aux_outputs['one_hot'] = one_hot_aux
                aux_outputs['text_mask'] = outputs['text_mask']
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)                
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = []
            for j in range(len(cat_list)): # bs
                interm_output_single = {
                    'pred_logits' : interm_outputs['pred_logits'][j].unsqueeze(0),
                    'pred_boxes': interm_outputs['pred_boxes'][j].unsqueeze(0)
                }
                inds = self.matcher(interm_output_single, [targets[j]], label_map_list[j])
                indices.extend(inds)
            one_hot_aux = torch.zeros(outputs['pred_logits'].size(),dtype=torch.int64)
            tgt_ids = [v["labels"].cpu() for v in targets]
            for i in range(len(indices)):
                tgt_ids[i]=tgt_ids[i][indices[i][1]]
                one_hot_aux[i,indices[i][0]] = label_map_list[i][tgt_ids[i]].to(torch.long)
            interm_outputs['one_hot'] = one_hot_aux
            interm_outputs['text_mask'] = outputs['text_mask']
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                kwargs = {}
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list


        return losses

def build_groundingdino(args, tokenizer, bert):  # vendored: no build registry; the tokenizer and BERT are passed in
    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
        tokenizer=tokenizer,  # vendored
        bert=bert,  # vendored
    )



    matcher = build_matcher(args)

    # prepare weight dict
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    

    clean_weight_dict = copy.deepcopy(weight_dict)

    # upstream to-do: this is a hack  # vendored: comment reworded (placeholder tokens are refused in the tutorial)
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update({k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)

    # losses = ['labels', 'boxes', 'cardinality']
    losses = ['labels', 'boxes']

    criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,losses=losses
                             )
    criterion.to(device)
    postprocessors = None  # vendored: PostProcess (COCO-style top-k boxes) is not carried; the pipeline thresholds the sigmoid scores as upstream's inference scripts do

    return model, criterion, postprocessors

def create_positive_map_exemplar(input_ids, label, special_tokens):
    tokens_positive = torch.zeros(256, dtype=torch.float)
    count = -1
    for token_ind in range(len(input_ids)):
        input_id = input_ids[token_ind]
        if (input_id not in special_tokens) and (token_ind == 0 or (input_ids[token_ind - 1] in special_tokens)):
            count += 1
        if count == label:
            ind_to_insert_ones = token_ind
            
            while input_ids[ind_to_insert_ones] not in special_tokens:
                tokens_positive[ind_to_insert_ones] = 1
                ind_to_insert_ones += 1
            break
    return tokens_positive
