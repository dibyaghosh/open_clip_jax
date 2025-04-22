from flax import linen as nn
from jaxtyping import Float, Int, Array
import einops
import math
import jax
from functools import partial
import jax.numpy as jnp

class functional:
    @staticmethod
    def linear(x: Float[Array, "... d"], weight: Float[Array, "dout din"], bias: Float[Array, "dout"] | None):
        out = x @ weight.T
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def normalize(x: Float[Array, "b l d"], axis: int = -1, eps: float = 1e-12):
        denom = jnp.clip(jnp.linalg.norm(x, axis=axis, keepdims=True), min=eps)
        return x / denom
    
    @staticmethod
    def layer_norm(x: Array, normalized_shape: tuple, weight: Array | None, bias: Array | None, eps: float = 1e-5):
        dtype = x.dtype
        x = x.astype(jnp.float32)
        reduction_axis = range(x.ndim - len(normalized_shape), x.ndim)
        mean = x.mean(axis=reduction_axis, keepdims=True)
        mean2 = jnp.mean(jnp.square(x), axis=reduction_axis, keepdims=True)
        var = jnp.maximum(0, mean2 - jnp.square(mean))
        y = x - mean
        mul = jax.lax.rsqrt(var + eps)
        if weight is not None:
            mul = mul * weight
        y = y * mul
        if bias is not None:
            y = y + bias
        return y.astype(dtype)

F = functional

class Linear(nn.Module):
    in_features: int
    out_features: int
    bias: bool = True

    def setup(self):
        scale = 1 / math.sqrt(self.in_features)
        initializer = lambda key, shape: jax.random.uniform(key, shape, minval=-1 * scale, maxval=scale)
        self.weight = self.param("weight", initializer, (self.out_features, self.in_features))
        if self.bias:
            self._bias = self.param("bias", nn.initializers.zeros, (self.out_features,))
        else:
            self._bias = None
    
    def __call__(self, x: Float[Array, "b n d"]):
        return F.linear(x, self.weight, self._bias)

class LayerNorm(nn.Module):
    normalized_shape: tuple | int
    eps: float = 1e-5
    elementwise_affine: bool = True
    bias: bool = True

    @property
    def shape(self):
        if isinstance(self.normalized_shape, int):
            return (self.normalized_shape,)
        else:
            return self.normalized_shape

    def setup(self):
        self.weight = self.param("weight", nn.initializers.ones, self.shape)
        if self.bias:
            self._bias = self.param("bias", nn.initializers.zeros, self.shape)
        else:
            self._bias = None
        
    def __call__(self, x: Float[Array, "b n d"]):
        return F.layer_norm(x, self.shape, self.weight, self._bias, self.eps)

class Identity(nn.Module):
    def __call__(self, x: Float[Array, "..."]):
        return x


class MultiheadAttention(nn.Module):
    embed_dim: int
    num_heads: int
    dropout: float = 0.
    bias: bool = True
    add_bias_kv: bool = False
    add_zero_attn: bool = False

    def setup(self):
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.in_proj_weight = self.param("in_proj_weight", nn.initializers.xavier_uniform(), (self.embed_dim * 3, self.embed_dim))
        if self.bias:
            self.in_proj_bias = self.param("in_proj_bias", nn.initializers.zeros, (self.embed_dim * 3,))
        else:
            self.in_proj_bias = None
        self.attn_drop = nn.Dropout(self.dropout)
        self.out_proj = Linear(self.embed_dim, self.embed_dim, bias=self.bias)
        assert not self.add_bias_kv, "Not implemented"
    
    def __call__(self, x, attn_mask: Array | None = None, *, train: bool):
        n, l, c = x.shape
        q, k, v = jnp.split(F.linear(x, self.in_proj_weight, self.in_proj_bias), 3, axis=-1)
        q = einops.rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = einops.rearrange(k, "b l (h d) -> b h l d", h=self.num_heads)
        v = einops.rearrange(v, "b l (h d) -> b h l d", h=self.num_heads)
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k)
        attn = attn * self.scale
        if attn_mask is not None:
            attn = jnp.where(attn_mask, attn, jnp.finfo(attn.dtype).min)
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, deterministic=not train)
        x = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        x = einops.rearrange(x, "b h l d -> b l (h d)")
        x = self.out_proj(x)
        return x

gelu = partial(nn.gelu, approximate=False)
        
class PatchConv(nn.Module):
    patch_size: int
    output_dim: int
    input_dim: int
    bias: bool

    def setup(self):
        scale = 1 / math.sqrt(self.patch_size * self.patch_size * self.input_dim)
        initializer = lambda key, shape: jax.random.uniform(key, shape, minval=-1 * scale, maxval=scale)
        self.weight = self.param("weight", initializer, (self.output_dim, self.input_dim, self.patch_size, self.patch_size))
        if self.bias:
            self._bias = self.param("bias", nn.initializers.zeros, (self.output_dim,))
        else:
            self._bias = None

    def __call__(self, x: Float[Array, "b h w c"]):
        x = einops.rearrange(x, "b (h ph) (w pw) c -> b h w (ph pw c)", ph=self.patch_size, pw=self.patch_size)
        return F.linear(x, einops.rearrange(self.weight, "o i ph pw -> o (ph pw i)"), self._bias)