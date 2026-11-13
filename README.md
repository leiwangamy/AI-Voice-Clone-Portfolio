# AI Voice Cloning — CosyVoice3 Fine-Tuning Engineering

Lei Wang

Notes on configuring, testing, and checking a CosyVoice3 LLM fine-tune. The model architecture and training algorithm are FunAudioLLM/CosyVoice. The work shown here is the engineering around that library: epoch setup, a formal-training lock, a T4 16 GB batch adjustment, and checkpoint checks.

This repository is a short public portfolio for AI, machine learning, and Python roles. It is not a voice-cloning product.

## What is included

`notebooks/AI-Voice-Clone-Portfolio.ipynb` has five short sections. Each one states the problem, shows a small Python snippet from the real configuration work, and states the limit of that snippet. The cells that compare batch settings and training locks can be run with a normal Python install. They do not need a GPU, a recording, or model weights.

## Workflow that was configured

1. Pin CosyVoice to commit `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` and continue from the official `llm.pt` in `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`. Flow and HiFi-GAN stay at the released weights.
2. `train.py` has no epoch flag. `max_epoch` is written into a copy of the official yaml as `start_epoch + 1 + epochs`. Learning rate stays at the official `1e-5` with `constantlr`.
3. A one-epoch smoke run on 8 clips has to pass before formal training. Formal training also waits for a typed positive epoch count and an explicit confirmation flag.
4. On an NVIDIA T4 16 GB GPU, the official formal batch `max_frames_in_batch` 2000 with `accum_grad` 2 did not fit the first batch. The run used 1000 and 4. The nominal product is still 4000. This is a memory setting, not a claim that the training curve matched the official batch.
5. After training, the checkpoint file is loaded and checked: step greater than zero, tensor names and shapes aligned with `init.pt`, at least one tensor changed, no non-finite values, and no extra epoch file. `init.pt` alone is not treated as success.

The upstream entry used for the real run is the same stage-5 `torchrun` path as CosyVoice `run.sh`: one GPU, fp32, `torch_ddp`, `--model llm`. This portfolio does not include that command or a copy of `train.py`.

## Engineering in this portfolio

- Pinning the library commit and the official LLM checkpoint.
- Writing `max_epoch` because the trainer has no epoch argument.
- Keeping formal training behind a smoke pass and a separate confirmation flag.
- Reducing frames per step on a 16 GB T4 while keeping the same nominal accumulated frame count.
- Checking a finished checkpoint instead of treating `init.pt` as a trained model.

## Privacy and limits

This repository does not contain training data, personal recordings, fine-tuned weights, checkpoint files, or a complete runnable voice-cloning application. Snippets that use example numbers are marked as illustrative. No loss curve or evaluation score is reported here.

Original model and trainer: [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice), Apache License 2.0. See `THIRD_PARTY_NOTICES.md`.

Research and engineering notes only. Not a product, and not a claim of authorship over CosyVoice3.
