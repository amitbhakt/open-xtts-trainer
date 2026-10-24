"""The only place model constructors are called. Defaults come from SPEC; overrides exist for small test models."""

from typing import Optional

from xtts_trainer.spec.xtts_spec import SPEC


def build_dvae(
    num_tokens: Optional[int] = None,
    codebook_dim: Optional[int] = None,
    hidden_dim: Optional[int] = None,
):
    """Native XTTS-v2 DiscreteVAE, built exactly as upstream GPTTrainer builds it."""
    from TTS.tts.layers.xtts.dvae import DiscreteVAE

    arch = SPEC.dvae
    return DiscreteVAE(
        channels=arch.channels,
        normalization=None,
        positional_dims=arch.positional_dims,
        num_tokens=num_tokens or arch.num_tokens,
        codebook_dim=codebook_dim or arch.codebook_dim,
        hidden_dim=hidden_dim or arch.hidden_dim,
        num_resnet_blocks=arch.num_resnet_blocks,
        kernel_size=arch.kernel_size,
        num_layers=arch.num_layers,
        use_transposed_convs=arch.use_transposed_convs,
    )


def build_gpt(
    number_text_tokens: int,
    start_text_token: int = 261,
    stop_text_token: int = 0,
    num_audio_tokens: Optional[int] = None,
    layers: Optional[int] = None,
    model_dim: Optional[int] = None,
    heads: Optional[int] = None,
    use_perceiver_resampler: Optional[bool] = None,
):
    """Native XTTS-v2 GPT, built as Xtts.init_models builds it from model_args.

    Text token ids must come from the tokenizer ([START]/[STOP]); the defaults are the stock-vocab ids.
    Audio start/stop tokens are the last two ids of num_audio_tokens, matching the DVAE codebook size.
    """
    from TTS.tts.layers.xtts.gpt import GPT

    arch = SPEC.gpt
    num_audio_tokens = num_audio_tokens or arch.num_audio_tokens
    return GPT(
        layers=layers or arch.layers,
        model_dim=model_dim or arch.model_dim,
        start_text_token=start_text_token,
        stop_text_token=stop_text_token,
        heads=heads or arch.heads,
        max_text_tokens=arch.max_text_tokens,
        max_mel_tokens=arch.max_audio_tokens,
        max_prompt_tokens=arch.max_prompt_tokens,
        number_text_tokens=number_text_tokens,
        num_audio_tokens=num_audio_tokens,
        start_audio_token=num_audio_tokens - 2,
        stop_audio_token=num_audio_tokens - 1,
        use_perceiver_resampler=arch.use_perceiver_resampler if use_perceiver_resampler is None else use_perceiver_resampler,
        code_stride_len=arch.code_stride_len,
    )


def build_hifigan():
    """Native XTTS-v2 HifiDecoder, built as Xtts.init_models builds it from model_args."""
    from TTS.tts.layers.xtts.hifigan_decoder import HifiDecoder

    return HifiDecoder(
        input_sample_rate=SPEC.gpt_sample_rate,
        output_sample_rate=SPEC.output_sample_rate,
        output_hop_length=SPEC.output_hop_length,
        ar_mel_length_compression=SPEC.gpt.code_stride_len,
        decoder_input_dim=SPEC.decoder_input_dim,
        d_vector_dim=SPEC.d_vector_dim,
        cond_d_vector_in_each_upsampling_layer=SPEC.cond_d_vector_in_each_upsampling_layer,
    )


def hifigan_decoder_input(decoder, latents):
    """HifiDecoder.forward's input stage on its own: GPT latents [B, T, C] -> decoder frames [B, C, F].

    Mirrors the two F.interpolate calls of HifiDecoder.forward, keeping the batch dimension. The generator
    (decoder.waveform_decoder) turns each frame into exactly output_hop_length samples, so interpolating a
    whole clip first and slicing frames afterwards keeps audio and latents aligned by construction.
    """
    import torch.nn.functional as F

    z = F.interpolate(latents.transpose(1, 2),
                      scale_factor=[decoder.ar_mel_length_compression / decoder.output_hop_length], mode="linear")
    if decoder.output_sample_rate != decoder.input_sample_rate:
        z = F.interpolate(z, scale_factor=[decoder.output_sample_rate / decoder.input_sample_rate], mode="linear")
    return z

