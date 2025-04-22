from flax import linen as nn
from jaxtyping import Float, Int, Array
from typing import Callable, Optional, Tuple, Union, List
import einops
import math
import jax
import torch_nn
from torch_nn import functional as F
from functools import partial
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
import flax



class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = True
    scaled_cosine: bool = False
    scale_heads: bool = False
    logit_scale_max: float = math.log(1. / 0.01)
    batch_first: bool = True
    attn_drop: float = 0.
    proj_drop: float = 0.

    def setup(self):
        self.scale = self.head_dim ** -0.5
        self.in_proj_weight = self.param(
            "in_proj_weight",
            nn.initializers.normal(self.scale),
            (self.dim * 3, self.dim),
        )
        if self.qkv_bias:
            self.in_proj_bias = self.param("in_proj_bias", nn.initializers.zeros, (self.dim * 3,))
        else:
            self.in_proj_bias = None
        

        if self.scaled_cosine:
            self.logit_scale = self.param("logit_scale", nn.initializers.constant(math.log(10)), (self.num_heads, 1, 1))
        else:
            self.logit_scale = None
        
        self.attn_drop = nn.Dropout(self.attn_drop)
        if self.scale_heads:
            self.head_scale = self.param("head_scale", nn.initializers.ones, (self.num_heads, 1, 1))
        else:
            self.head_scale = 1
        self.out_proj = torch_nn.Linear(self.dim, self.dim)
        self.out_drop = nn.Dropout(self.proj_drop)
    
    def forward(self, x: Float[Array, "b n d"], attn_mask: Float[Array, "b 1 l l"] | None = None):
        n, l, c = x.shape
        q, k, v = jnp.split(F.linear(x, self.in_proj_weight, self.in_proj_bias), 3, axis=-1)
        q = einops.rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = einops.rearrange(k, "b l (h d) -> b h l d", h=self.num_heads)
        v = einops.rearrange(v, "b l (h d) -> b h l d", h=self.num_heads)

        if self.logit_scale is not None:
            q, k = F.normalize(q), F.normalize(k)

        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k)
        if self.logit_scale is not None:
            attn = attn * jnp.exp(jnp.clip(self.logit_scale, max=self.logit_scale_max))
        else:
            attn = attn * self.scale
        if attn_mask is not None:
            attn = jnp.where(attn_mask, attn, jnp.finfo(attn.dtype).min)
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        if self.head_scale is not None:
            x = x * self.head_scale

        x = einops.rearrange(x, "b h l d -> b l (h d)")
        x = self.out_proj(self.out_drop(x))
        return x

class LayerScale(nn.Module):
    shape: tuple
    init_value: float = 1e-5

    def setup(self):
        self.gamma = self.param("gamma", nn.initializers.constant(self.init_value), self.shape)

    def __call__(self, x: Float[Array, "b n d"]):
        return x * self.gamma

class TransformerMLP(nn.Module):
    width: int
    mlp_ratio: float
    act_layer: Callable = torch_nn.gelu

    def setup(self):
        mlp_width = int(self.width * self.mlp_ratio)
        self.c_fc = torch_nn.Linear(self.width, mlp_width)
        self.c_proj = torch_nn.Linear(mlp_width, self.width)
    

    def __call__(self, x: Float[Array, "b n d"]):
        x = self.c_fc(x)
        x = self.act_layer(x)
        x = self.c_proj(x)
        return x

class ResidualAttentionBlock(nn.Module):
    d_model: int
    n_head: int
    mlp_ratio: float
    ls_init_value: float = None
    act_layer: Callable = torch_nn.gelu
    norm_layer: Callable = torch_nn.LayerNorm
    is_cross_attention: bool = False

    def setup(self):
        self.ln_1 = self.norm_layer(self.d_model)
        self.attn = torch_nn.MultiheadAttention(self.d_model, self.n_head)
        if self.ls_init_value is not None:
            self.ls_1 = LayerScale(self.d_model, self.ls_init_value)
            self.ls_2 = LayerScale(self.d_model, self.ls_init_value)
        else:
            self.ls_1 = torch_nn.Identity()
            self.ls_2 = torch_nn.Identity()
        
        self.ln_2 = self.norm_layer(self.d_model)
        self.mlp = TransformerMLP(self.d_model, self.mlp_ratio, self.act_layer)

        if self.is_cross_attention:
            self.ln_1_kv = norm_layer(self.d_model)
    
    def __call__(self, x: Float[Array, "b l d"], attn_mask: Float[Array, "b 1 l l"] | None = None, *, train: bool):
        x = x + self.ls_1(self.attn(self.ln_1(x), attn_mask=attn_mask, train=train))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x

class Transformer(nn.Module):
    width: int
    layers: int
    heads: int
    mlp_ratio: float
    ls_init_value: float = None
    act_layer: Callable = torch_nn.gelu
    norm_layer: Callable = nn.LayerNorm

    def setup(self):
        self.resblocks = [
            ResidualAttentionBlock(self.width, self.heads, self.mlp_ratio, self.ls_init_value, self.act_layer, self.norm_layer, is_cross_attention=False)
            for _ in range(self.layers)
        ]
    
    def __call__(self, x: Float[Array, "b l d"], *, attn_mask: Float[Array, "b 1 l l"] | None = None, train: bool):
        for block in self.resblocks:
            x = block(x, attn_mask=attn_mask, train=train)
        return x

class PatchDropout(nn.Module):
    dropout: float
    has_cls_token: bool = True

    @nn.compact
    def __call__(self, x: Float[Array, "b l d"], *, train: bool):
        if not train:
            return x
        cls_token, x = jnp.split(x, [int(self.has_cls_token)], axis=1)
        tokens_to_keep = max(1, int(x.shape[1] * (1 - self.dropout)))
        key = self.make_rng("dropout")
        rand = jax.random.uniform(key, (x.shape[0], x.shape[1]))        
        patch_indices_keep = jax.lax.top_k(rand, tokens_to_keep)[1]
        x = x[patch_indices_keep]
        return jnp.concatenate([cls_token, x], axis=1)
        

class VisionTransformer(nn.Module):
    image_size: int
    patch_size: int
    width: int
    layers: int
    heads: int
    mlp_ratio: float
    ls_init_value: float = None
    attentional_pool: bool = False
    attn_pooler_queries: int = 256
    attn_pooler_heads: int = 8
    output_dim: int = 512
    patch_dropout: float = 0.
    no_ln_pre: bool = False
    pos_embed_type: str = 'learnable'
    pool_type: str = 'tok'
    final_ln_after_pool: bool = False
    act_layer: Callable = torch_nn.gelu
    norm_layer: Callable = torch_nn.LayerNorm
    output_tokens: bool = False
    image_normalization_mean: tuple = (0.48145466, 0.4578275, 0.40821073)
    image_normalization_std: tuple = (0.26862954, 0.26130258, 0.27577711)

    def setup(self):
        grid_size = (self.image_size // self.patch_size, self.image_size // self.patch_size)
        scale = self.width ** -0.5
        self.conv1 = torch_nn.PatchConv(
            patch_size=self.patch_size,
            output_dim=self.width,
            input_dim=3,
            bias=False,
        )
        self.class_embedding = self.param(
            "class_embedding",
            nn.initializers.normal(scale),
            (self.width,),
        )
        if self.pos_embed_type == 'learnable':
            self.positional_embedding = self.param(
                "positional_embedding",
                nn.initializers.normal(scale),
                (grid_size[0] * grid_size[1] + 1, self.width),
            )   
        elif self.pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert grid_size[0] == grid_size[1],\
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = self.param(
                "positional_embedding",
                nn.initializers.normal(scale),
                (grid_size[0] * grid_size[1] + 1, self.width),
            )
        else:
            raise ValueError
        
        self.transformer = Transformer(self.width, self.layers, self.heads, self.mlp_ratio, self.ls_init_value, self.act_layer, self.norm_layer)
        self.ln_post = self.norm_layer(self.width)
        self.proj = self.param("proj", nn.initializers.normal(scale), (self.width, self.output_dim))
        self.patch_dropout_module = PatchDropout(self.patch_dropout)
        self.ln_pre = self.norm_layer(self.width) if not self.no_ln_pre else torch_nn.Identity()

    def _embeds(self, image: Int[Array, "b h w c"], train: bool) -> Float[Array, "b n d"]:
        image = image.astype(jnp.float32) / 255.0
        image = image - jnp.asarray(self.image_normalization_mean)
        image = image / jnp.asarray(self.image_normalization_std)

        x = self.conv1(image)
        x = einops.rearrange(x, "b ph pw d -> b (ph pw) d")

        # class embeddings and positional embeddings
        class_embedding = jnp.broadcast_to(self.class_embedding, (x.shape[0], 1, x.shape[-1]))
        x = jnp.concatenate([class_embedding, x], axis=1)
        x = x + self.positional_embedding

        # patch dropout (if active)
        x = self.patch_dropout_module(x, train=train)
        x = self.ln_pre(x)
        return x

    def _global_pool(self, x: Float[Array, "b n d"]) -> Tuple[Float[Array, "b d"], Float[Array, "b phpw d"]]:
        if self.pool_type == 'avg':
            return x[:, 1:].mean(axis=1), x[:, 1:]
        elif self.pool_type == 'tok':
            return x[:, 0], x[:, 1:]
        else:
            raise ValueError
            
    def _pool(self, x: Float[Array, "b n d"]) -> Tuple[Float[Array, "b d"], Float[Array, "b phpw d"]]:
        if self.attentional_pool:
            raise NotImplementedError()
        if self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
        return pooled, tokens
        
    def __call__(self, image: Int[Array, "b h w c"], *, output_tokens: bool = False, train: bool) -> Float[Array, "b d"]:
        x = self._embeds(image, train=train)
        x = self.transformer(x, train=train)
        pooled, tokens = self._pool(x)
        if self.proj is not None:
            pooled = pooled @ self.proj
        if self.output_tokens:
            return pooled, tokens
        return pooled

class TorchEmbed(nn.Embed):
    def setup(self):
        self.embedding = self.param(
        'weight',
        self.embedding_init,
        (self.num_embeddings, self.features),
        self.param_dtype,
        )


class TextTransformer(nn.Module):
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: float = None
    output_dim: int = 512
    embed_cls: bool = False
    no_causal_mask: bool = False
    pad_id: int = 0
    pool_type: str = 'argmax'
    proj_type: str = 'linear'
    proj_bias: bool = False
    act_layer: Callable = torch_nn.gelu
    norm_layer: Callable = torch_nn.LayerNorm
    output_tokens: bool = False

    def setup(self):
        self.token_embedding = TorchEmbed(self.vocab_size, self.width, embedding_init=nn.initializers.normal(0.02))
        if self.embed_cls:
            self.cls_emb = self.param("cls_emb", nn.initializers.normal(0.01), (self.width,))
            self.num_pos += 1
        else:
            self.cls_emb = None

        self.num_pos = self.context_length + int(bool(self.embed_cls))
        self.positional_embedding = self.param("positional_embedding", nn.initializers.normal(0.01), (self.num_pos, self.width))
        self.transformer = Transformer(self.width, self.layers, self.heads, self.mlp_ratio, self.ls_init_value, self.act_layer, self.norm_layer)

        self.ln_final = self.norm_layer(self.width)
        if self.proj_type == 'none' or not self.output_dim:
            self.text_projection = None
        else:
            if self.proj_bias:
                self.text_projection = torch_nn.Linear(self.width, self.output_dim)
            else:
                scale = self.width ** -0.5
                self.text_projection = self.param("text_projection", nn.initializers.normal(scale), (self.width, self.output_dim))


    def _embeds(self, text: Int[Array, "b n_tok"]) -> tuple[Float[Array, "b l d"], Float[Array, "b 1 l l"]]:
        x = self.token_embedding(text)
        valid_mask = (text != self.pad_id)

        if self.embed_cls:
            x = jnp.concatenate([x, jnp.broadcast_to(self.cls_emb, (x.shape[0], 1, x.shape[-1]))], axis=1)
            valid_mask = jnp.pad(valid_mask, ((0, 0), (0, 1)), constant_values=True)

        valid_attn_mask = jnp.logical_and(valid_mask[:, None, :], valid_mask[:, :, None])
        causal_mask = jnp.arange(x.shape[1])[:, None] >= jnp.arange(x.shape[1])[None, :]
        attn_mask = jnp.logical_and(valid_attn_mask, causal_mask)

        x = x + self.positional_embedding[:x.shape[1]]
        return x, attn_mask[:, None, :, :]
    
    def __call__(self, text: Int[Array, "b n_tok"], *, train: bool, output_tokens: bool = False) -> Float[Array, "b d"]:
        x, attn_mask = self._embeds(text)
        x = self.transformer(x, attn_mask=attn_mask, train=train)
        if self.embed_cls:
            pooled = x[:, -1]
            pooled = self.ln_final(pooled)
            tokens = x[:, :-1]
        else:
            x = self.ln_final(x)
            pooled = text_global_pool(x, text, pool_type=self.pool_type)
        
        if self.text_projection is not None:
            if self.proj_bias:
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens
        return pooled


def text_global_pool(x: Float[Array, "b l d"], text: Int[Array, "b n_tok"], pool_type: str) -> Float[Array, "b d"]:
    if pool_type == 'first':
        return x[:, 0]
    elif pool_type == 'last':
        return x[:, -1]
    elif pool_type == 'argmax':
        return x[jnp.arange(x.shape[0]), text.argmax(axis=-1)]
    else:
        raise ValueError

@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = 'learnable'
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'tok'
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None

    timm_model_name: Optional[str] = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth

    def build(vision_cfg, embed_dim: int, quick_gelu: bool = False):
        act_layer = torch_nn.gelu
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = torch_nn.LayerNorm
        return VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )




@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None  # layer scale initial value
    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False  # disable causal masking
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'argmax'
    proj_bias: bool = False
    proj_type: str = 'linear'  # control final text projection, 'none' forces no projection
    output_tokens: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None

    # HuggingFace specific text tower config
    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = 'mlp'
    hf_pooler_type: str = 'mean_pooler'  # attentional pooling for HF models

    def build(text_cfg, embed_dim: int, quick_gelu: bool = False):
        act_layer = torch_nn.gelu
        norm_layer = torch_nn.LayerNorm
        return TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            mlp_ratio=text_cfg.mlp_ratio,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            no_causal_mask=text_cfg.no_causal_mask,
            pad_id=text_cfg.pad_id,
            pool_type=text_cfg.pool_type,
            proj_type=text_cfg.proj_type,
            proj_bias=text_cfg.proj_bias,
            output_tokens=text_cfg.output_tokens,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

@dataclass
class CLIPCfg:
    embed_dim: int
    vision_cfg: dict | CLIPVisionCfg
    text_cfg: dict | CLIPTextCfg
    quick_gelu: bool = False
    init_logit_scale: float = np.log(1 / 0.07)

    def build(self):
        vision_cfg = self.vision_cfg
        if not isinstance(vision_cfg, CLIPVisionCfg):
            vision_cfg = CLIPVisionCfg(**vision_cfg)
        text_cfg = self.text_cfg
        if not isinstance(text_cfg, CLIPTextCfg):
            text_cfg = CLIPTextCfg(**text_cfg)
        return CLIP(self.embed_dim, vision_cfg, text_cfg, self.quick_gelu, self.init_logit_scale)


class CLIP(nn.Module):
    embed_dim: int
    vision_cfg: CLIPVisionCfg
    text_cfg: CLIPTextCfg
    quick_gelu: bool = False
    init_logit_scale: float = np.log(1 / 0.07)

    def setup(self):
        self.visual = self.vision_cfg.build(self.embed_dim, self.quick_gelu)
        self.text = self.text_cfg.build(self.embed_dim, self.quick_gelu)
        nn.share_scope(self, self.text)
        self.logit_scale = self.param("logit_scale", nn.initializers.constant(self.init_logit_scale), tuple())
    
    def encode_image(self, image, train: bool, normalize: bool = False):
        phi = self.visual(image, train=train, output_tokens=False)
        if normalize:
            phi = F.normalize(phi, axis=-1)
        return phi
    
    def encode_text(self, text, train: bool, normalize: bool = False):
        psi = self.text(text, train=train, output_tokens=False)
        if normalize:
            psi = F.normalize(psi, axis=-1)
        return psi

    def __call__(self, image, text, train: bool):
        image_features = self.encode_image(image, train, normalize=True)
        text_features = self.encode_text(text, train, normalize=True)
        logit_scale = jnp.exp(self.logit_scale)
        return logit_scale * image_features @ text_features.T

def reformat_params(flat_torch_params: dict):
    params = {k.replace('resblocks.', 'resblocks_'): v for k, v in flat_torch_params.items()}
    return flax.traverse_util.unflatten_dict(params, sep='.')