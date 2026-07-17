# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paddle implementation of the Moondream2 vision-language architecture."""

import math
from typing import Optional, Tuple, Union

import paddle
from paddle import nn
from paddle.distributed.fleet.utils import recompute
from paddle.nn import functional as F

from ..cache_utils import Cache, DynamicCache
from ..model_outputs import CausalLMOutputWithPast
from ..model_utils import PretrainedModel
from .configuration import Moondream2Config, Moondream2TextConfig, Moondream2VisionConfig

__all__ = [
    "Moondream2ForConditionalGeneration",
    "Moondream2PreTrainedModel",
    "Moondream2VisionModel",
    "build_moondream2_prefix_mask",
]


def gelu_tanh(hidden_states):
    return F.gelu(hidden_states, approximate=True)


def build_moondream2_prefix_mask(query_length, key_value_length, prefix_length, past_length=0, dtype="float32"):
    """Build Moondream2's explicit additive attention mask.

    The first ``prefix_length`` positions are mutually visible.  Every later
    query can attend causally to all prior positions.  The returned tensor has
    shape ``[1, 1, query_length, key_value_length]`` and is suitable for eager
    attention scores.
    """
    query_positions = paddle.arange(past_length, past_length + query_length).unsqueeze(-1)
    key_positions = paddle.arange(key_value_length).unsqueeze(0)
    visible = key_positions <= query_positions
    prefix_visible = (query_positions < prefix_length) & (key_positions < prefix_length)
    visible = visible | prefix_visible
    zero = paddle.zeros([query_length, key_value_length], dtype=dtype)
    negative = paddle.full([query_length, key_value_length], -1e9, dtype=dtype)
    return paddle.where(visible, zero, negative).unsqueeze(0).unsqueeze(0)


def apply_partial_rope(query, key, position_ids, rotary_dim, theta):
    """Apply Moondream2's non-interleaved RoPE to only the leading dimensions."""
    if rotary_dim == 0:
        return query, key
    inv_freq = 1.0 / (theta ** (paddle.arange(0, rotary_dim, 2, dtype="float32") / rotary_dim))
    positions = position_ids.astype("float32").unsqueeze(-1)
    freqs = positions * inv_freq.unsqueeze(0).unsqueeze(0)
    cos = paddle.cos(freqs).unsqueeze(1)
    sin = paddle.sin(freqs).unsqueeze(1)

    def rotate(states):
        rotated, passthrough = states[..., :rotary_dim], states[..., rotary_dim:]
        half = rotary_dim // 2
        # Match the source implementation exactly: calculate the complex
        # rotation in FP32, then cast the concatenated rotated portion back to
        # the projection dtype.  Paddle previously downcast the trig factors
        # before multiplication, which changes the BF16 inference path.
        real = rotated[..., :half].astype("float32")
        imag = rotated[..., half:].astype("float32")
        out = paddle.concat([real * cos - imag * sin, real * sin + imag * cos], axis=-1).astype(states.dtype)
        return paddle.concat([out, passthrough], axis=-1)

    return rotate(query), rotate(key)


class Moondream2MLP(nn.Layer):
    def __init__(self, in_features, intermediate_size, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, out_features)

    def forward(self, hidden_states):
        return self.fc2(gelu_tanh(self.fc1(hidden_states)))


class Moondream2VisionAttention(nn.Layer):
    def __init__(self, config: Moondream2VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        batch_size, sequence_length, _ = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape([batch_size, sequence_length, 3, self.num_heads, self.head_dim])
        query, key, value = qkv.unbind(axis=2)
        query, key, value = query.transpose([0, 2, 1, 3]), key.transpose([0, 2, 1, 3]), value.transpose([0, 2, 1, 3])
        scores = paddle.matmul(query * self.scale, key, transpose_y=True)
        attention = F.softmax(scores, axis=-1)
        output = paddle.matmul(attention, value).transpose([0, 2, 1, 3]).reshape([batch_size, sequence_length, -1])
        return self.proj(output)


class Moondream2VisionBlock(nn.Layer):
    def __init__(self, config: Moondream2VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.attn = Moondream2VisionAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.mlp = Moondream2MLP(config.hidden_size, config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Moondream2VisionModel(nn.Layer):
    """ViT used for every global/local Moondream image crop."""

    def __init__(self, config: Moondream2VisionConfig):
        super().__init__()
        self.config = config
        self.grid_size = config.crop_size // config.patch_size
        self.patch_embed = nn.Linear(config.num_channels * config.patch_size * config.patch_size, config.hidden_size)
        self.position_embedding = self.create_parameter(
            [1, self.grid_size * self.grid_size, config.hidden_size], default_initializer=nn.initializer.Normal(std=0.02)
        )
        self.blocks = nn.LayerList([Moondream2VisionBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def forward(self, pixel_values):
        batch_size, _, height, width = pixel_values.shape
        if height != self.config.crop_size or width != self.config.crop_size:
            raise ValueError(f"Moondream2 expects {self.config.crop_size}x{self.config.crop_size} crops, got {height}x{width}")
        patch = self.config.patch_size
        grid = self.grid_size
        patches = pixel_values.reshape([batch_size, self.config.num_channels, grid, patch, grid, patch])
        patches = patches.transpose([0, 2, 4, 1, 3, 5]).reshape([batch_size, grid * grid, -1])
        hidden_states = self.patch_embed(patches) + self.position_embedding
        for block in self.blocks:
            # A native image contains one global and up to twelve local crops.
            # Storing every ViT block's activations for all crops exceeds a 46 GB
            # GPU during full FP32 SFT, although inference itself fits normally.
            # Recompute each vision block in backward to retain only its input.
            # This changes memory/runtime trade-offs only while training and
            # leaves the eager inference/parity path untouched.
            if self.training:
                hidden_states = recompute(block, hidden_states)
            else:
                hidden_states = block(hidden_states)
        return self.norm(hidden_states)


class Moondream2VisionProjector(nn.Layer):
    def __init__(self, config: Moondream2VisionConfig):
        super().__init__()
        self.mlp = Moondream2MLP(config.hidden_size * 2, config.projection_inner_dim, config.projection_dim)

    def forward(self, global_features, reconstructed_features):
        return self.mlp(paddle.concat([global_features, reconstructed_features], axis=-1))


class Moondream2TextAttention(nn.Layer):
    def __init__(self, config: Moondream2TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_dim = config.rotary_dim
        self.rope_theta = config.rope_theta
        self.scale = self.head_dim**-0.5
        projection_size = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
        self.qkv = nn.Linear(config.hidden_size, projection_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, position_ids, attention_mask, past_key_values=None, use_cache=False):
        batch_size, query_length, _ = hidden_states.shape
        qkv = self.qkv(hidden_states)
        query_end = self.num_heads * self.head_dim
        key_end = query_end + self.num_key_value_heads * self.head_dim
        query = qkv[..., :query_end].reshape([batch_size, query_length, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        key = qkv[..., query_end:key_end].reshape([batch_size, query_length, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])
        value = qkv[..., key_end:].reshape([batch_size, query_length, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])
        query, key = apply_partial_rope(query, key, position_ids, self.rotary_dim, self.rope_theta)

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            key, value = past_key_values.update(key, value, self.layer_idx)

        if self.num_heads != self.num_key_value_heads:
            repeat = self.num_heads // self.num_key_value_heads
            key = key.repeat_interleave(repeat, axis=1)
            value = value.repeat_interleave(repeat, axis=1)
        scores = paddle.matmul(query * self.scale, key, transpose_y=True) + attention_mask.astype(query.dtype)
        attention = F.softmax(scores, axis=-1)
        output = paddle.matmul(attention, value).transpose([0, 2, 1, 3]).reshape([batch_size, query_length, -1])
        return self.out_proj(output), past_key_values


class Moondream2DecoderBlock(nn.Layer):
    """Parallel pre-norm residual block used by the original Moondream decoder."""

    def __init__(self, config: Moondream2TextConfig, layer_idx: int):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.attn = Moondream2TextAttention(config, layer_idx)
        self.mlp = Moondream2MLP(config.hidden_size, config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states, position_ids, attention_mask, past_key_values=None, use_cache=False):
        normalized = self.norm(hidden_states)
        attention_output, past_key_values = self.attn(
            normalized, position_ids, attention_mask, past_key_values=past_key_values, use_cache=use_cache
        )
        return hidden_states + attention_output + self.mlp(normalized), past_key_values


class Moondream2TextModel(nn.Layer):
    def __init__(self, config: Moondream2TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.LayerList([Moondream2DecoderBlock(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=False,
        attention_mask=None,
        prefix_length=0,
    ):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be specified")
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, query_length, _ = inputs_embeds.shape
        past_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        key_value_length = past_length + query_length
        position_ids = paddle.arange(past_length, key_value_length, dtype="int64").unsqueeze(0).expand([batch_size, -1])
        prefix_mask = build_moondream2_prefix_mask(
            query_length, key_value_length, prefix_length, past_length, dtype=inputs_embeds.dtype
        )
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError("Moondream2 attention_mask must have shape [batch, sequence]")
            key_padding = attention_mask[:, None, None, :key_value_length].astype(inputs_embeds.dtype)
            prefix_mask = prefix_mask + (1.0 - key_padding) * -1e9
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states, past_key_values = layer(
                hidden_states, position_ids, prefix_mask, past_key_values=past_key_values, use_cache=use_cache
            )
        return self.norm(hidden_states), past_key_values


class Moondream2RegionModel(nn.Layer):
    """Optional coordinate/size embeddings and decoders for detect/point tasks."""

    def __init__(self, config):
        super().__init__()
        # The public checkpoint stores Fourier frequencies as [input_dim,
        # feature_dim / 2]: scalar coordinates use [1, 128] and sizes use
        # [2, 256].  This preserves its exact tensor layout.
        self.coordinate_features = self.create_parameter(
            [1, config.coordinate_feature_dim // 2], default_initializer=nn.initializer.Normal(std=1.0)
        )
        self.size_features = self.create_parameter(
            [2, config.size_feature_dim // 2], default_initializer=nn.initializer.Normal(std=1.0)
        )
        self.coordinate_encoder = nn.Linear(config.coordinate_feature_dim, config.hidden_size)
        self.size_encoder = nn.Linear(config.size_feature_dim, config.hidden_size)
        self.coordinate_decoder = Moondream2MLP(config.hidden_size, config.intermediate_size, config.coordinate_output_dim)
        self.size_decoder = Moondream2MLP(config.hidden_size, config.intermediate_size, config.size_output_dim)

    @staticmethod
    def _fourier(values, weights):
        frequencies = 2.0 * math.pi * paddle.matmul(values, weights)
        return paddle.concat([paddle.cos(frequencies), paddle.sin(frequencies)], axis=-1)

    def encode_coordinates(self, coordinates):
        return self.coordinate_encoder(self._fourier(coordinates, self.coordinate_features))

    def encode_sizes(self, sizes):
        return self.size_encoder(self._fourier(sizes, self.size_features))

    def decode_coordinates(self, hidden_states):
        return self.coordinate_decoder(hidden_states)

    def decode_sizes(self, hidden_states):
        return self.size_decoder(hidden_states).reshape([*hidden_states.shape[:-1], 2, -1])


class Moondream2PreTrainedModel(PretrainedModel):
    config_class = Moondream2Config

    @classmethod
    def _gen_aoa_config(cls, config):
        """Describe the identity mapping used by PaddleFormers flex checkpoints."""
        return {"aoa_statements": []}

    @classmethod
    def _gen_inv_aoa_config(cls, config):
        """Describe the inverse identity mapping used when saving flex checkpoints."""
        return {"aoa_statements": []}
    base_model_prefix = "model"
    _no_split_modules = ["Moondream2DecoderBlock", "Moondream2VisionBlock"]
    _keep_in_fp32_modules = ["model.norm"]

    def _init_weights(self, layer):
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            if getattr(layer, "weight", None) is not None:
                nn.initializer.Normal(std=0.02)(layer.weight)
            if getattr(layer, "bias", None) is not None:
                nn.initializer.Constant(0.0)(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            nn.initializer.Constant(1.0)(layer.weight)
            nn.initializer.Constant(0.0)(layer.bias)


class Moondream2ForConditionalGeneration(Moondream2PreTrainedModel):
    """Moondream2 model with image-prefix injection and causal language head."""

    def __init__(self, config: Moondream2Config):
        super().__init__(config)
        self.config = config
        self.vision_model = Moondream2VisionModel(config.vision_config)
        self.vision_projector = Moondream2VisionProjector(config.vision_config)
        self.model = Moondream2TextModel(config.text_config)
        self.region_model = Moondream2RegionModel(config.region_config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size)
        # Strict FP64 inference is only required by the dedicated cross-framework
        # logit-parity harness.  It must stay opt-in: SFT configurations that
        # request FP32 need actual FP32 parameters, optimizer states and kernels.
        self._strict_logits_alignment = False

    def to(self, *args, **kwargs):
        """Optionally promote the dedicated logit-parity harness to FP64."""
        requested_dtype = kwargs.get("dtype")
        if self._strict_logits_alignment and requested_dtype in ("float32", paddle.float32):
            kwargs["dtype"] = "float64"
        return super().to(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def _reconstruct_local_features(self, crop_features, tiles_h, tiles_w):
        """Stitch local ViT features using the source overlap-boundary semantics."""
        grid = self.vision_model.grid_size
        margin = self.config.vision_config.overlap_margin
        tiles_h, tiles_w = int(tiles_h), int(tiles_w)
        if tiles_h * tiles_w != crop_features.shape[0]:
            raise ValueError(
                f"Local crop count ({crop_features.shape[0]}) does not match tiling ({tiles_h}, {tiles_w})"
            )

        # Match ``reconstruct_from_crops`` in the native implementation: discard
        # overlap only on edges shared with another local crop.  In particular a
        # 1x1 tiling must preserve all 27x27 features, rather than cropping it to
        # the 19x19 interior and upsampling it again.
        rows = []
        index = 0
        for tile_y in range(tiles_h):
            cells = []
            y_start = 0 if tile_y == 0 else margin
            y_end = grid if tile_y == tiles_h - 1 else grid - margin
            for tile_x in range(tiles_w):
                x_start = 0 if tile_x == 0 else margin
                x_end = grid if tile_x == tiles_w - 1 else grid - margin
                cell = crop_features[index].reshape([grid, grid, -1])
                cells.append(cell[y_start:y_end, x_start:x_end])
                index += 1
            rows.append(paddle.concat(cells, axis=1))
        reconstructed = paddle.concat(rows, axis=0).transpose([2, 0, 1]).unsqueeze(0)
        reconstructed = F.adaptive_avg_pool2d(reconstructed, output_size=[grid, grid])
        return reconstructed.squeeze(0).transpose([1, 2, 0]).reshape([grid * grid, -1])

    def encode_image(self, pixel_values, image_sizes=None, crop_counts=None):
        """Encode normalized global/local crops into the fixed 729-token image prefix."""
        pixel_values = pixel_values.astype(self.vision_model.patch_embed.weight.dtype)
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)
        if pixel_values.ndim != 5:
            raise ValueError("pixel_values must have shape [batch, crops, channels, height, width]")
        batch_size, max_crops = pixel_values.shape[:2]
        if crop_counts is None:
            crop_counts = paddle.full([batch_size], max_crops, dtype="int64")
        outputs = []
        for batch_idx in range(batch_size):
            count = int(crop_counts[batch_idx])
            encoded = self.vision_model(pixel_values[batch_idx, :count])
            global_features = encoded[0]
            if count == 1:
                reconstructed = global_features
            else:
                if image_sizes is None:
                    tiles_h, tiles_w = 1, count - 1
                else:
                    tiles_h, tiles_w = int(image_sizes[batch_idx, 0]), int(image_sizes[batch_idx, 1])
                reconstructed = self._reconstruct_local_features(encoded[1:], tiles_h, tiles_w)
            outputs.append(self.vision_projector(global_features, reconstructed))
        return paddle.stack(outputs, axis=0)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        crop_counts=None,
        use_cache=True,
        **kwargs,
    ):
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            past_length = past_key_values.get_seq_length()
            input_ids = input_ids[:, -1:]
            inputs_embeds = None
            pixel_values = None
            image_sizes = None
            crop_counts = None

            # GenerationMixin extends the caller-provided text mask by one token
            # per decode step, but the cache also contains the BOS + image prefix.
            # Restore those omitted prefix columns before applying the cached
            # attention mask.
            if attention_mask is not None:
                key_value_length = past_length + input_ids.shape[1]
                missing_prefix = key_value_length - attention_mask.shape[-1]
                if missing_prefix > 0:
                    prefix_attention = paddle.ones(
                        [attention_mask.shape[0], missing_prefix],
                        dtype=attention_mask.dtype,
                    )
                    attention_mask = paddle.concat(
                        [prefix_attention, attention_mask], axis=-1
                    )
        return {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
            "crop_counts": crop_counts,
            "use_cache": use_cache,
        }

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        image_sizes: Optional[paddle.Tensor] = None,
        crop_counts: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        use_cache = self.config.use_cache if use_cache is None else use_cache
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

        # PaddleFormers' generic VL collator supplies a binary causal mask with
        # shape [batch, 1, query, key].  Moondream2 constructs its own causal
        # and image-prefix mask below, so it only needs the corresponding 2-D
        # key-padding mask.  Reducing over heads and queries preserves real
        # token columns (including a final one-token sequence) and excludes
        # padded columns before the image prefix is prepended.
        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask = paddle.max(attention_mask, axis=[1, 2])

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be specified")
            inputs_embeds = self.model.embed_tokens(input_ids)
        else:
            inputs_embeds = inputs_embeds.astype(self.model.embed_tokens.weight.dtype)
        has_image_prefix = pixel_values is not None
        if has_image_prefix:
            image_features = self.encode_image(pixel_values, image_sizes=image_sizes, crop_counts=crop_counts)
            # The native image path pre-fills a bidirectional 730-token prefix:
            # one BOS embedding followed by the 729 projected image tokens.  The
            # BOS must be part of the cached prefix rather than the text prompt,
            # otherwise image generation positions and attention visibility are
            # shifted by one token relative to the reference implementation.
            bos_ids = paddle.full([inputs_embeds.shape[0], 1], self.config.bos_token_id, dtype="int64")
            bos_features = self.model.embed_tokens(bos_ids).astype(inputs_embeds.dtype)
            prefix_features = paddle.concat([bos_features, image_features.astype(inputs_embeds.dtype)], axis=1)
            inputs_embeds = paddle.concat([prefix_features, inputs_embeds], axis=1)
            if attention_mask is not None:
                prefix_attention = paddle.ones([attention_mask.shape[0], prefix_features.shape[1]], dtype=attention_mask.dtype)
                attention_mask = paddle.concat([prefix_attention, attention_mask], axis=1)

        past_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        # Native text-only queries use a regular causal mask. The special
        # bidirectional prefix is active only for image sequences and their
        # subsequent cached decode steps.
        prefix_length = (
            self.config.text_config.prefix_attn
            if has_image_prefix or past_length >= self.config.text_config.prefix_attn
            else 0
        )
        hidden_states, past_key_values = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
            prefix_length=prefix_length,
        )
        logits = self.lm_head(hidden_states).astype("float32")
        loss = None
        if labels is not None:
            if pixel_values is not None:
                prefix_labels = paddle.full([labels.shape[0], logits.shape[1] - labels.shape[1]], -100, dtype=labels.dtype)
                labels = paddle.concat([prefix_labels, labels], axis=1)
            shift_logits = logits[:, :-1].reshape([-1, logits.shape[-1]])
            shift_labels = labels[:, 1:].reshape([-1])
            # Paddle's ``cross_entropy(..., ignore_index=-100, reduction="mean")``
            # zeros ignored targets but divides by every flattened position. PyTorch
            # instead divides by non-ignored targets, which is the causal-SFT
            # contract used by ms-swift. Compute the per-token NLL then normalize
            # explicitly by supervised-token count so text-only and image-prefix
            # batches share the PyTorch/HF loss scale.
            token_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="none").reshape([-1])
            supervised = (shift_labels != -100).astype(token_loss.dtype)
            supervised_count = paddle.clip(paddle.sum(supervised), min=1.0)
            loss = paddle.sum(token_loss * supervised) / supervised_count
        if not return_dict:
            output = (logits, past_key_values)
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values)
