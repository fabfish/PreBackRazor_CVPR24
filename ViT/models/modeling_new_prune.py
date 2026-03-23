# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn

import torch.nn.functional as F

import numpy as np
# from models.custom_functions.sparse_matrix import SparseTensor

from custom_functions.custom_fc import LinearSparse
from custom_functions.custom_matmul import MatMulSparse
from custom_functions.custom_softmax_matmul import SoftmaxMatMulSparse

from torch.nn import Dropout, Softmax

from pdb import set_trace

import mesa as ms

from typing import Optional, Union

# from flash_attn import flash_attn_func
# Based on the FlashAttention v2 interface code.

# import flash_attn_cuda
import flash_attn_2_cuda as flash_attn_cuda
# flash_attn version 2.2.0

import sys
import logging

from mesa import custom_quant
from mesa import native
from mesa import packbit

from custom_functions.sparse_matrix import sparsify, unsparsify

class Masker(object):
    def __init__(self, prune_ratio):
        self.prune_ratio = prune_ratio

    @torch.no_grad()
    def __call__(self, activation):
        num_small = int(np.clip(activation[0].numel() * self.prune_ratio, 1, activation[0].numel()))
        activation_mag = torch.abs(activation)
        threshold, _ = torch.kthvalue(activation_mag.flatten(1), num_small)
        while len(threshold.shape) < len(activation_mag.shape):
            threshold = threshold.unsqueeze(-1)
        mask = activation_mag >= threshold

        # print("mask density is {}".format(mask.float().mean()))
        # idle mask
        # mask = torch.ones_like(activation).to(torch.bool)

        return mask

class MlpActPrune(nn.Module):
    def __init__(self, config, masker, prebackrazor):
        super(MlpActPrune, self).__init__()

        self.fc1 = LinearSparse(config.hidden_size, config.transformer["mlp_dim"], quantize=config.quantize, half=config.half, masker=masker, prebackrazor=prebackrazor)
        self.fc2 = LinearSparse(config.transformer["mlp_dim"], config.hidden_size, quantize=config.quantize, half=config.half, masker=masker, prebackrazor=prebackrazor)

        self.num_attention_heads = config.transformer["num_heads"]
        self.act_fn = nn.GELU()

        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class AttentionActPrune(nn.Module):
    def __init__(self, config, vis, masker, prebackrazor, flashattn):
        super(AttentionActPrune, self).__init__()
        self.vis = vis
        self.masker = masker
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = LinearSparse(config.hidden_size, self.all_head_size, quantize=config.quantize, half=config.half, masker=masker, prebackrazor=prebackrazor)
        self.key = LinearSparse(config.hidden_size, self.all_head_size, quantize=config.quantize, half=config.half, masker=masker, prebackrazor=prebackrazor)
        self.value = LinearSparse(config.hidden_size, self.all_head_size, quantize=config.quantize, half=config.half, masker=masker, prebackrazor=prebackrazor)

        self.out = LinearSparse(config.hidden_size, config.hidden_size, quantize=config.quantize, half=config.half, masker=masker, prebackrazor=prebackrazor)

        assert config.transformer["attention_dropout_rate"] == 0
        # self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.flashattn = flashattn
        self.prebackrazor = prebackrazor
        if flashattn is False:
            self.mm1 = MatMulSparse(quantize=config.quantize, half=config.half, masker=masker, prebackrazor=prebackrazor)
            # self.softmax_mm2 = SoftmaxMatMulSparse(quantize=config.quantize, masker=masker, dim=-1)
            # self.mm2 = MatMulSparse(quantize=config.quantize, half=config.half, masker=masker, prebackrazor=prebackrazor)

            self.softmax_mm2 = SoftmaxMatMulSparse(dim=-1, quantize=config.quantize, half=config.half, masker=masker, prebackrazor=prebackrazor)
            self.mm2 = None
            
            # self.attn_kernel = None
            self.attn_masker = None
        else:
            # self.attn_kernel = BackRazorFlashAttention(quantize=config.quantize, half=config.half, masker=masker)
            self.attn_masker = Masker(0.8)
            # self.attn_kernel = 


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        if self.flashattn:
            return x
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        """
        Notes:
        - This path currently targets the FlashAttention CUDA kernels.
        - The dq/dk/dv computation is not skipped in this configuration.
        - We still apply sparsification to the q/k/v activations to reduce saved backward memory.
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # if self.flashattn is False:
        #     query_layer = self.transpose_for_scores(mixed_query_layer)
        #     key_layer = self.transpose_for_scores(mixed_key_layer)
        #     value_layer = self.transpose_for_scores(mixed_value_layer)
        # else:
        #     query_layer = self.transpose_for_scores(mixed_query_layer).to(torch.float16)
        #     key_layer = self.transpose_for_scores(mixed_key_layer).to(torch.float16)
        #     value_layer = self.transpose_for_scores(mixed_value_layer).to(torch.float16)
        
        # query_layer = self.transpose_for_scores(mixed_query_layer)
        # key_layer = self.transpose_for_scores(mixed_key_layer)
        # value_layer = self.transpose_for_scores(mixed_value_layer)
        
        weights = None
        if self.flashattn is False:
            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)
            
            attention_scores = self.mm1(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            # context_layer = self.softmax_mm2(attention_scores, value_layer)

            if self.mm2 is None:
                context_layer = self.softmax_mm2(attention_scores, value_layer)
            else:
                attention_probs = self.softmax(attention_scores)
                context_layer = self.mm2(attention_probs, value_layer)
                
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            
                
        else:
            # query_layer = mixed_query_layer.to(torch.float16)
            # key_layer = mixed_key_layer.to(torch.float16)
            # value_layer = mixed_value_layer.to(torch.float16)
            
            q_requires_grad = self.query.weight.requires_grad or (self.query.bias is not None and self.query.bias.requires_grad)
            k_requires_grad = self.key.weight.requires_grad or (self.key.bias is not None and self.key.bias.requires_grad)
            v_requires_grad = self.value.weight.requires_grad or (self.value.bias is not None and self.value.bias.requires_grad)
            
            # if not q_requires_grad:
            #     query_layer = query_layer.detach()
            # if not k_requires_grad:
            #     key_layer = key_layer.detach()
            # if not v_requires_grad:
            #     value_layer = value_layer.detach()
            
            ###############################################################
            # One line to enable flash attention cuda kernel (fwd + bwd)
            ###############################################################
            # context_layer = flash_attn_func(query_layer, key_layer, value_layer, dropout_p=0.0, softmax_scale=None, causal=False).to(torch.float32)
            
            
            
            ###############################################################
            # BackRazor FlashAttention Kernel
            ###############################################################
            
            if not (q_requires_grad or k_requires_grad or v_requires_grad) or not self.training:
                query_layer = self.transpose_for_scores(mixed_query_layer).to(torch.float16).detach()
                key_layer = self.transpose_for_scores(mixed_key_layer).to(torch.float16).detach()
                value_layer = self.transpose_for_scores(mixed_value_layer).to(torch.float16).detach()
                # query_layer = mixed_query_layer.to(torch.float16).detach()
                # key_layer = mixed_key_layer.to(torch.float16).detach()
                # value_layer = mixed_value_layer.to(torch.float16).detach()
                context_layer = flash_attn_fwd_func(query_layer, key_layer, value_layer, dropout_p=0.0, softmax_scale=None, causal=False).to(torch.float32)
                
            else:
                
                dice = torch.rand(1)
                
                if dice > 0.7:
                    
                    ############################################################
                    # Using Original Flash_Func (No Backrazor)
                    ############################################################
                    query_layer = self.transpose_for_scores(mixed_query_layer).to(torch.float16)
                    key_layer = self.transpose_for_scores(mixed_key_layer).to(torch.float16)
                    value_layer = self.transpose_for_scores(mixed_value_layer).to(torch.float16)
                    if not q_requires_grad:
                        query_layer = query_layer.detach()
                    if not k_requires_grad:
                        key_layer = key_layer.detach()
                    if not v_requires_grad:
                        value_layer = value_layer.detach()
                    # context_layer = self.attn_kernel(query_layer, key_layer, value_layer).to(torch.float32)
                    context_layer = flash_attn_func(query_layer, key_layer, value_layer).to(torch.float32)
                    
                else:
                
                    ############################################################
                    # Using BackRazor Flash_Func
                    ############################################################
                    query_layer = self.transpose_for_scores(mixed_query_layer)
                    key_layer = self.transpose_for_scores(mixed_key_layer)
                    value_layer = self.transpose_for_scores(mixed_value_layer)
                    mask_q = self.attn_masker(query_layer)
                    mask_k = self.attn_masker(key_layer)
                    mask_v = self.attn_masker(value_layer)
                    # context_layer = easy_backrazor_flash_attn_func(query_layer, key_layer, value_layer)

                    # Recompute-based variant (integrated with selective sparse backward flow).
                    context_layer = recompute_backrazor_flash_attn_func(query_layer, key_layer, value_layer, mask_q, mask_k, mask_v)
                
                # To test no recomputation version, use
                # context_layer = lazy_backrazor_flash_attn_func(query_layer, key_layer, value_layer, mask_q, mask_k, mask_v)
                
                
            
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


"""
This section wraps FlashAttention interfaces so they can integrate with selective sparse backprop.
We keep the interface reference for context.
"""

def _flash_attn_forward(q, k, v, dropout_p, softmax_scale, causal, return_softmax):
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = flash_attn_cuda.fwd(
        q, k, v, None, dropout_p, softmax_scale, causal, return_softmax, None
    )
    return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state

def _flash_attn_backward(
    dout, q, k, v, out, softmax_lse, dq, dk, dv, dropout_p, softmax_scale, causal, rng_state=None
):
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    dq, dk, dv, softmax_d, = flash_attn_cuda.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        dropout_p,
        softmax_scale,
        causal,
        None,
        rng_state,
    )
    return dq, dk, dv, softmax_d

class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dropout_p, softmax_scale, causal, return_softmax):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            return_softmax=return_softmax and dropout_p > 0,
        )
        ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None
    
def flash_attn_func(
    q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, return_attn_probs=False
):
    return FlashAttnFunc.apply(q, k, v, dropout_p, softmax_scale, causal, return_attn_probs)

####################################################
# BackRazor-based implementation/wrapper for FlashAttention.
####################################################

class FlashAttnFwdFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dropout_p, softmax_scale, causal, return_softmax):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            return_softmax=return_softmax and dropout_p > 0,
        )
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        assert False, "Not Supposed To Do FlashAttn BWD, Please Check"
        return None, None, None, None, None, None, None, None, None, None, None
    
def flash_attn_fwd_func(
    q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, return_attn_probs=False
):
    return FlashAttnFwdFunc.apply(q, k, v, dropout_p, softmax_scale, causal, return_attn_probs)

#########################################################################################################################
# FlashAttention wrappers (kept for selective sparse-backprop integration)
#########################################################################################################################

class LazyBackRazorFlashAttnFunc(torch.autograd.Function):
    '''
    Lazy recompute wrapper for FlashAttention integrated with sparse backward storage.

    Input:
      - q, k, v (float32)
      - mask_q, mask_k, mask_v (bool masks)
    Output:
      - out (float32)
    '''
    @staticmethod
    def forward(ctx, q, k, v, mask_q, mask_k, mask_v, dropout_p, softmax_scale, causal, return_softmax):
        if softmax_scale is None:
            softmax_scale = q.to(torch.float16).shape[-1] ** (-0.5)
            
        shape_x_1, mask_x_1, sparse_x_1 = sparsify(q, mask_q)
        shape_x_2, mask_x_2, sparse_x_2 = sparsify(k, mask_k)
        shape_x_3, mask_x_3, sparse_x_3 = sparsify(v, mask_v)
        
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            q.to(torch.float16),
            k.to(torch.float16),
            v.to(torch.float16),
            dropout_p,
            softmax_scale,
            causal=causal,
            return_softmax=return_softmax and dropout_p > 0,
        )
        
        
        
        # ctx.save_for_backward(shape_x_1, mask_x_1, sparse_x_1, shape_x_2, mask_x_2, sparse_x_2, shape_x_3, mask_x_3, sparse_x_3)
        ctx.save_for_backward(shape_x_1, mask_x_1, sparse_x_1, shape_x_2, mask_x_2, sparse_x_2, shape_x_3, mask_x_3, sparse_x_3, out, softmax_lse, rng_state)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        
        out = out.to(torch.float32)
        
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        # shape_x_1, mask_x_1, sparse_x_1, shape_x_2, mask_x_2, sparse_x_2, shape_x_3, mask_x_3, sparse_x_3 = ctx.saved_tensors
        shape_x_1, mask_x_1, sparse_x_1, shape_x_2, mask_x_2, sparse_x_2, shape_x_3, mask_x_3, sparse_x_3, out, softmax_lse, rng_state = ctx.saved_tensors
        
        q = unsparsify(shape_x_1, mask_x_1, sparse_x_1).to(torch.float16)
        k = unsparsify(shape_x_2, mask_x_2, sparse_x_2).to(torch.float16)
        v = unsparsify(shape_x_3, mask_x_3, sparse_x_3).to(torch.float16)

        
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        dqr, dkr, dvr = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        
        
        _flash_attn_backward(
            dout.to(torch.float16),
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            rng_state=rng_state,
        )
        
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            q.to(torch.float16),
            k.to(torch.float16),
            v.to(torch.float16),
            ctx.dropout_p,
            ctx.softmax_scale,
            causal=ctx.causal,
            return_softmax=False,
        )
        
        _flash_attn_backward(
            dout.to(torch.float16),
            q,
            k,
            v,
            out,
            softmax_lse,
            dqr,
            dkr,
            dvr,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            rng_state=rng_state,
        )
        
        if torch.norm(dq-dqr)>0.001 or torch.norm(dk-dkr)>0.001 or torch.norm((dv-dvr))>0.001 :
            print(torch.norm(dq-dqr),torch.norm(dk-dkr),torch.norm((dv-dvr)))
        
        # import pdb;pdb.set_trace()
        
        dq = dq[..., : dout.shape[-1]].to(torch.float32)  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]].to(torch.float32)
        dv = dv[..., : dout.shape[-1]].to(torch.float32)
        
        if dq.isnan().any().item() or dk.isnan().any().item() or dv.isnan().any().item():
            print("Q gradient:", "NaNs!" if dq.isnan().any().item() else "OK")
            print("K gradient:", "NaNs!" if dk.isnan().any().item() else "OK")
            print("V gradient:", "NaNs!" if dv.isnan().any().item() else "OK")
        
        return dq, dk, dv, None, None, None, None, None, None, None, None
    
def lazy_backrazor_flash_attn_func(
    q, k, v, mask_q, mask_k, mask_v, dropout_p=0.0, softmax_scale=None, causal=False, return_attn_probs=False
):
    return LazyBackRazorFlashAttnFunc.apply(q, k, v, mask_q, mask_k, mask_v, dropout_p, softmax_scale, causal, return_attn_probs)

# Recompute-based variant (fallback that works with selective sparse backprop).

class RecomputeBackRazorFlashAttnFunc(torch.autograd.Function):
    """
    Recompute-based FlashAttention wrapper.

    Inputs:
      - q, k, v (float32)
      - mask_q, mask_k, mask_v (bool masks for sparsification)
    Output:
      - out (float32)
    """
    @staticmethod
    def forward(ctx, q, k, v, mask_q, mask_k, mask_v, dropout_p, softmax_scale, causal, return_softmax):
        if softmax_scale is None:
            softmax_scale = q.to(torch.float16).shape[-1] ** (-0.5)
            
        shape_x_1, mask_x_1, sparse_x_1 = sparsify(q, mask_q)
        shape_x_2, mask_x_2, sparse_x_2 = sparsify(k, mask_k)
        shape_x_3, mask_x_3, sparse_x_3 = sparsify(v, mask_v)
        
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            q.to(torch.float16),
            k.to(torch.float16),
            v.to(torch.float16),
            dropout_p,
            softmax_scale,
            causal=causal,
            return_softmax=return_softmax and dropout_p > 0,
        )
        
        out = out.to(torch.float32)
        
        ctx.save_for_backward(shape_x_1, mask_x_1, sparse_x_1, shape_x_2, mask_x_2, sparse_x_2, shape_x_3, mask_x_3, sparse_x_3)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        shape_x_1, mask_x_1, sparse_x_1, shape_x_2, mask_x_2, sparse_x_2, shape_x_3, mask_x_3, sparse_x_3 = ctx.saved_tensors
        
        q = unsparsify(shape_x_1, mask_x_1, sparse_x_1)
        k = unsparsify(shape_x_2, mask_x_2, sparse_x_2)
        v = unsparsify(shape_x_3, mask_x_3, sparse_x_3)
        
        out, q, k, v, _, softmax_lse, _, rng_state = _flash_attn_forward(
            q.to(torch.float16),
            k.to(torch.float16),
            v.to(torch.float16),
            ctx.dropout_p,
            ctx.softmax_scale,
            causal=ctx.causal,
            return_softmax=False,
        )
        
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        
        _flash_attn_backward(
            dout.to(torch.float16),
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            rng_state=rng_state,
        )
        
        dq = dq[..., : dout.shape[-1]].to(torch.float32)  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]].to(torch.float32)
        dv = dv[..., : dout.shape[-1]].to(torch.float32)
        
        return dq, dk, dv, None, None, None, None, None, None, None, None
    
def recompute_backrazor_flash_attn_func(
    q, k, v, mask_q, mask_k, mask_v, dropout_p=0.0, softmax_scale=None, causal=False, return_attn_probs=False
):
    return RecomputeBackRazorFlashAttnFunc.apply(q, k, v, mask_q, mask_k, mask_v, dropout_p, softmax_scale, causal, return_attn_probs)

#########################################################################################################################
# Easier variant that avoids custom quantization in the wrapper.
#########################################################################################################################

class EasyBackRazorFlashAttnFunc(torch.autograd.Function):
    """
    Easier recompute wrapper variant without custom quantization.

    Inputs:
      - q, k, v (float32)
      - mask_q, mask_k, mask_v (bool masks for sparsification)
    Output:
      - out (float32)
    """
    @staticmethod
    def forward(ctx, q, k, v, mask_q, mask_k, mask_v, dropout_p, softmax_scale, causal, return_softmax):
        if softmax_scale is None:
            softmax_scale = q.to(torch.float16).shape[-1] ** (-0.5)
            
        shape_x_1, mask_x_1, sparse_x_1 = sparsify(q, mask_q)
        shape_x_2, mask_x_2, sparse_x_2 = sparsify(k, mask_k)
        shape_x_3, mask_x_3, sparse_x_3 = sparsify(v, mask_v)
        
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            q.to(torch.float16),
            k.to(torch.float16),
            v.to(torch.float16),
            dropout_p,
            softmax_scale,
            causal=causal,
            return_softmax=return_softmax and dropout_p > 0,
        )
        
        out = out.to(torch.float32)
        
        ctx.save_for_backward(shape_x_1, mask_x_1, sparse_x_1, shape_x_2, mask_x_2, sparse_x_2, shape_x_3, mask_x_3, sparse_x_3, out_padded, softmax_lse, rng_state)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        dout = dout.to(torch.float16)
        shape_x_1, mask_x_1, sparse_x_1, shape_x_2, mask_x_2, sparse_x_2, shape_x_3, mask_x_3, sparse_x_3, out, softmax_lse, rng_state = ctx.saved_tensors
        
        q = unsparsify(shape_x_1, mask_x_1, sparse_x_1).to(torch.float16)
        k = unsparsify(shape_x_2, mask_x_2, sparse_x_2).to(torch.float16)
        v = unsparsify(shape_x_3, mask_x_3, sparse_x_3).to(torch.float16)
        
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]].to(torch.float32)  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]].to(torch.float32)
        dv = dv[..., : dout.shape[-1]].to(torch.float32)
        return dq, dk, dv, None, None, None, None, None, None, None, None
    
def easy_backrazor_flash_attn_func(
    q, k, v, mask_q, mask_k, mask_v, dropout_p=0.0, softmax_scale=None, causal=False, return_attn_probs=False
):
    return EasyBackRazorFlashAttnFunc.apply(q, k, v, mask_q, mask_k, mask_v, dropout_p, softmax_scale, causal, return_attn_probs)

#########################################################################################################################

class backrazorflashattention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, input3, mask1, mask2, mask3, quantize, half, 
                dropout_p=0.0, softmax_scale=None, causal=False, return_softmax=False,
                clip_val1=None, level1=256, iteration1=None, ema_decay1=None, quant_groups1=None, shift1=None,
                clip_val2=None, level2=256, iteration2=None, ema_decay2=None, quant_groups2=None, shift2=None,
                clip_val3=None, level3=256, iteration3=None, ema_decay3=None, quant_groups3=None, shift3=None):
        
        maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
        input1, input2, input3 = [maybe_contiguous(x) for x in (input1, input2, input3)]

        shape_x_1, mask_x_1, sparse_x_1 = sparsify(input1, mask1)
        shape_x_2, mask_x_2, sparse_x_2 = sparsify(input2, mask2)
        shape_x_3, mask_x_3, sparse_x_3 = sparsify(input3, mask3)

        if half and (not quantize):
            sparse_x_1 = sparse_x_1.half()
            sparse_x_2 = sparse_x_2.half()
            sparse_x_3 = sparse_x_3.half()

        if softmax_scale is None:
            softmax_scale = input1.shape[-1] ** (-0.5)
            
        out, _, _, _, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            input1.to(torch.float16),
            input2.to(torch.float16),
            input3.to(torch.float16),
            dropout_p,
            softmax_scale,
            causal=causal,
            return_softmax=return_softmax and dropout_p > 0,
        )
        
        if quantize:
            custom_quant.Quant.forward(ctx, sparse_x_1, clip_val1, level1, iteration1, ema_decay1, quant_groups1, shift1, '_1')
            custom_quant.Quant.forward(ctx, sparse_x_2, clip_val2, level2, iteration2, ema_decay2, quant_groups2, shift2, '_2')
            custom_quant.Quant.forward(ctx, sparse_x_3, clip_val3, level3, iteration3, ema_decay3, quant_groups3, shift3, '_3')

            ctx.save_for_backward(shape_x_1, shape_x_2, shape_x_3, mask_x_1, mask_x_2, mask_x_3,
                                  out_padded, softmax_lse, rng_state)
        else:
            ctx.save_for_backward(shape_x_1, shape_x_2, shape_x_3, mask_x_1, mask_x_2, mask_x_3, sparse_x_1, sparse_x_2, sparse_x_3,
                                  out_padded, softmax_lse, rng_state)
        
        # ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
        
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, grad_output, *args):

        tensors = ctx.saved_tensors
        if len(tensors) == 9:
            shape_x_1, shape_x_2, shape_x_3, mask_x_1, mask_x_2, mask_x_3, out_padded, softmax_lse, rng_state = tensors

            sparse_x_1 = custom_quant.Quant.restore(ctx, '_1')
            sparse_x_2 = custom_quant.Quant.restore(ctx, '_2')
            sparse_x_3 = custom_quant.Quant.restore(ctx, '_3')
            
        else:
            shape_x_1, shape_x_2, shape_x_3, mask_x_1, mask_x_2, mask_x_3, sparse_x_1, sparse_x_2, sparse_x_3, out_padded, softmax_lse, rng_state = tensors

        # sparse_x_1 = sparse_x_1.float16()
        # sparse_x_2 = sparse_x_2.float16()
        # sparse_x_3 = sparse_x_3.float16()

        q = unsparsify(shape_x_1, mask_x_1, sparse_x_1).to(torch.float16)
        k = unsparsify(shape_x_2, mask_x_2, sparse_x_2).to(torch.float16)
        v = unsparsify(shape_x_3, mask_x_3, sparse_x_3).to(torch.float16)

        # if ctx.needs_input_grad[0]:
        #     grad_input1 = grad_output.matmul(input2.transpose(-2, -1).to(dtype=grad_output.dtype))
        # if ctx.needs_input_grad[1]:
        #     grad_input2 = input1.transpose(-2, -1).to(dtype=grad_output.dtype).matmul(grad_output)
        
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        
        maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
        # dq, dk, dv are allocated by us so they should already be contiguous
        grad_output, q, k, v, out_padded = [maybe_contiguous(x) for x in (grad_output, q, k, v, out_padded)]
        
        _flash_attn_backward(
            grad_output,
            q,
            k,
            v,
            out_padded,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            rng_state=rng_state,
        )
        dq = dq[..., : grad_output.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : grad_output.shape[-1]]
        dv = dv[..., : grad_output.shape[-1]]
        # Return dq/dk/dv matching the original head dimension.
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

class BackRazorFlashAttention(nn.Module):
    def __init__(self, args=None, logger=None, quant_groups=1, masker=None, quantize=False, half=False, prebackrazor=False,
                dropout_p=0.0, softmax_scale=None, causal=False, return_attn_probs=False):
        super(BackRazorFlashAttention, self).__init__()
        self.quant1 = custom_quant.quantization(tag='backrazorflashattn-1', quant_groups=quant_groups)
        self.quant2 = custom_quant.quantization(tag='backrazorflashattn-2', quant_groups=quant_groups)
        self.quant3 = custom_quant.quantization(tag='backrazorflashattn-3', quant_groups=quant_groups)
        self.quantize = quantize
        self.half = half
        self.masker = masker
        self.tag = 'backrazorflashattn'

        self.prebackrazor = prebackrazor
        
        self.dropout_p = dropout_p
        self.softmax_scale=softmax_scale 
        self.causal=causal
        self.return_attn_probs=return_attn_probs

    def update_quantization_parameter(self, **parameters):
        self.quant1.update_quantization_parameter(**parameters)
        self.quant2.update_quantization_parameter(**parameters)
        self.quant3.update_quantization_parameter(**parameters)

    def forward(self, x1, x2, x3):
        if self.masker is not None or self.training:
            mask1 = self.masker(x1)
            mask2 = self.masker(x2)
            mask3 = self.masker(x3)

            y = backrazorflashattention.apply(x1, x2, x3, mask1, mask2, mask3, self.quantize, self.half,
                             self.dropout_p, self.softmax_scale, self.causal, self.return_attn_probs,
                             self.quant1.clip_val, self.quant1.level, self.quant1.iteration, self.quant1.ema_decay, self.quant1.quant_groups, self.quant1.shift,
                             self.quant2.clip_val, self.quant2.level, self.quant2.iteration, self.quant2.ema_decay, self.quant2.quant_groups, self.quant2.shift,
                             self.quant3.clip_val, self.quant3.level, self.quant3.iteration, self.quant3.ema_decay, self.quant3.quant_groups, self.quant3.shift)
        else:
            y = flash_attn_fwd_func(x1, x2, x3, self.dropout_p, self.softmax_scale, self.causal)
        return y

