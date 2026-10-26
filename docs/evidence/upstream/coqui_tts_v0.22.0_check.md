# Upstream findings in the original `coqui-ai/TTS` v0.22.0

Checked 2026-10-24 for `docs/recipe_design.md` §2. Every value below is command output, not typed in.

## Inputs

```
$ gh api repos/coqui-ai/TTS/git/ref/tags/v0.22.0 --jq '.object | [.type, .sha] | @tsv'
commit	fa28f99f1508b5b5366539b2149963edcb80ba62

$ gh api repos/coqui-ai/TTS --jq '[.archived, .pushed_at, .default_branch] | @tsv'
false	2024-08-16T12:07:14Z	dev

$ for f in TTS/tts/layers/xtts/dvae.py TTS/tts/layers/xtts/gpt.py TTS/tts/layers/xtts/trainer/gpt_trainer.py \
           TTS/tts/layers/xtts/trainer/dataset.py TTS/tts/layers/xtts/tokenizer.py; do
    curl -sSfL --retry 3 -o "$(basename $f)" "https://raw.githubusercontent.com/coqui-ai/TTS/fa28f99f1508b5b5366539b2149963edcb80ba62/$f"; done
$ shasum -a 256 *.py
6ebc52b0c5ad17b9392ec7d3e4b6aa63aff4f28ea641eac648e1aac9dc593aac  dataset.py
e076aab664a8843373098b19a2fe7adc7a469b064e1f887444cf18b69ed86344  dvae.py
0c1448c358360fd6e39b107bf01c37bf31b71d678e209ef1925d47ca3bc09f02  gpt.py
6325607649be01340205d5f342dbc7da4b8a47773f5e91e3482c02a3d1589855  gpt_trainer.py
106f347814b7bd61ed04f54b39c5349cd046ae1cfe075be139081097cb42617c  tokenizer.py
```

Files are fetched by the commit the tag resolves to, not by the tag (a tag can be moved). Rerun 2026-10-25 with the commit URLs: the same five SHA-256 values as the first run by tag, so the excerpts below are from `fa28f99f`.

`archived: false`: the repository is unmaintained (last push 2024-08-16), not formally archived.

## The lines each finding rests on

Printed with `sed -n "<start>,<end>p" <file>`. Files are under `TTS/tts/layers/xtts/` (`dataset.py` and
`gpt_trainer.py` under `TTS/tts/layers/xtts/trainer/`).

| Finding | Excerpt |
|---|---|
| F1: codebook overwritten by EMA in train mode | `dvae.py:103-116` |
| F2: perceiver path divides the prompt span by 256 | `gpt.py:412-416` |
| F2: mask applied after the per-sample loop, with its leftover `idx` | `gpt.py:533-546` |
| F4: unlisted languages raise | `tokenizer.py:630-644` |
| Non-strict `xtts_checkpoint` load | `gpt_trainer.py:82-85` |
| F6: random new rows, old last row copied to the new last row | `gpt_trainer.py:116-137` |
| Non-strict DVAE load | `gpt_trainer.py:184-187` |
| Half-clip prompt for short clips | `dataset.py:30-37` |
| Exceptions swallowed, sample 1 substituted | `dataset.py:150-166` |

```
### dvae.py:103-116
        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            if distributed.is_initialized() and distributed.get_world_size() > 1:
                distributed.all_reduce(embed_onehot_sum)
                distributed.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
### gpt.py:412-416
            for idx in range(cond_idxs.size(0)):
                if self.use_perceiver_resampler:
                    cond_idxs[idx] = cond_idxs[idx] // self.perceiver_cond_length_compression
                else:
                    cond_idxs[idx] = cond_idxs[idx] // self.code_stride_len
### gpt.py:533-546
        for idx, l in enumerate(code_lengths):
            mel_targets[idx, l + 1 :] = -1

        # check if stoptoken is in every row of mel_targets
        assert (mel_targets == self.stop_audio_token).sum() >= mel_targets.shape[
            0
        ], f" ❗ mel_targets does not contain stop token ({self.stop_audio_token}) in every row."

        # ignore the loss for the segment used for conditioning
        # coin flip for the segment to be ignored
        if cond_idxs is not None:
            cond_start = cond_idxs[idx, 0]
            cond_end = cond_idxs[idx, 1]
            mel_targets[idx, cond_start:cond_end] = -1
### tokenizer.py:630-644
    def preprocess_text(self, txt, lang):
        if lang in {"ar", "cs", "de", "en", "es", "fr", "hu", "it", "nl", "pl", "pt", "ru", "tr", "zh", "ko"}:
            txt = multilingual_cleaners(txt, lang)
            if lang == "zh":
                txt = chinese_transliterate(txt)
            if lang == "ko":
                txt = korean_transliterate(txt)
        elif lang == "ja":
            txt = japanese_cleaners(txt, self.katsu)
        elif lang == "hi":
            # @manmay will implement this
            txt = basic_cleaners(txt)
        else:
            raise NotImplementedError(f"Language '{lang}' is not supported.")
        return txt
### gpt_trainer.py:82-85

        if self.args.xtts_checkpoint:
            self.load_checkpoint(self.config, self.args.xtts_checkpoint, eval=False, strict=False)

### gpt_trainer.py:116-137
                # add new tokens to a linear layer (text_head)
                emb_g = gpt_checkpoint["text_embedding.weight"]
                new_row = torch.randn(num_new_tokens, emb_g.shape[1])
                start_token_row = emb_g[-1, :]
                emb_g = torch.cat([emb_g, new_row], axis=0)
                emb_g[-1, :] = start_token_row
                gpt_checkpoint["text_embedding.weight"] = emb_g

                # add new weights to the linear layer (text_head)
                text_head_weight = gpt_checkpoint["text_head.weight"]
                start_token_row = text_head_weight[-1, :]
                new_entry = torch.randn(num_new_tokens, self.xtts.gpt.text_head.weight.shape[1])
                text_head_weight = torch.cat([text_head_weight, new_entry], axis=0)
                text_head_weight[-1, :] = start_token_row
                gpt_checkpoint["text_head.weight"] = text_head_weight

                # add new biases to the linear layer (text_head)
                text_head_bias = gpt_checkpoint["text_head.bias"]
                start_token_row = text_head_bias[-1]
                new_bias_entry = torch.zeros(num_new_tokens)
                text_head_bias = torch.cat([text_head_bias, new_bias_entry], axis=0)
                text_head_bias[-1] = start_token_row
### gpt_trainer.py:184-187
        if self.args.dvae_checkpoint:
            dvae_checkpoint = torch.load(self.args.dvae_checkpoint, map_location=torch.device("cpu"))
            self.dvae.load_state_dict(dvae_checkpoint, strict=False)
            print(">> DVAE weights restored from:", self.args.dvae_checkpoint)
### dataset.py:30-37
        sample_length = int((min_sample_length + max_sample_length) / 2)
    else:
        sample_length = random.randint(min_sample_length, max_sample_length)
    gap = rel_clip.shape[-1] - sample_length
    if gap < 0:
        sample_length = rel_clip.shape[-1] // 2
    gap = rel_clip.shape[-1] - sample_length

### dataset.py:150-166

        # ignore samples that we already know that is not valid ones
        if sample_id in self.failed_samples:
            if self.debug_failures:
                print(f"Ignoring sample {sample['audio_file']} because it was already ignored before !!")
            # call get item again to get other sample
            return self[1]

        # try to load the sample, if fails added it to the failed samples list
        try:
            tseq, audiopath, wav, cond, cond_len, cond_idxs = self.load_item(sample)
        except:
            if self.debug_failures:
                print(f"error loading {sample['audio_file']} {sys.exc_info()}")
            self.failed_samples.add(sample_id)
            return self[1]
```

## F7: the checkpoint loader that rewrites `model.pth` (added 2026-10-26)

F7 is checked against both code bases, since the claim is that the fork introduced it:

```
$ curl -sSfL -o xtts_v0.22.0.py https://raw.githubusercontent.com/coqui-ai/TTS/fa28f99f1508b5b5366539b2149963edcb80ba62/TTS/tts/models/xtts.py
$ curl -sSfL -o xtts_fork.py https://raw.githubusercontent.com/idiap/coqui-ai-TTS/03339e5b790b5e16ea8867a142e8bb26ec646d94/TTS/tts/models/xtts.py
$ shasum -a 256 xtts_v0.22.0.py xtts_fork.py
2be596d0900a8ebf4868101ebab9b6bf80409af280877176df826c0d63db3aae  xtts_v0.22.0.py
dcd55c554fe81a9600e837526ca33b09b7d934dd0bb6d80bb9857159204c055e  xtts_fork.py
$ grep -n "UnpicklingError\|torch.save(" xtts_v0.22.0.py
(no output: exit 1)
$ grep -n "UnpicklingError\|torch.save(checkpoint, model_path)" xtts_fork.py
642:        except pickle.UnpicklingError:
651:                torch.save(checkpoint, model_path)
```

The fork file is byte-identical to the one installed from the pinned commit by `uv sync --frozen` (the
`python3.13` part of the path is the interpreter version and varies; rerun 2026-10-26):

```
$ cmp xtts_fork.py .venv/lib/python3.13/site-packages/TTS/tts/models/xtts.py; echo "exit $?"
exit 0
$ shasum -a 256 .venv/lib/python3.13/site-packages/TTS/tts/models/xtts.py
dcd55c554fe81a9600e837526ca33b09b7d934dd0bb6d80bb9857159204c055e  .venv/lib/python3.13/site-packages/TTS/tts/models/xtts.py
```

So the in-place rewrite exists only in the fork; v0.22.0 loads the checkpoint without writing to it.

