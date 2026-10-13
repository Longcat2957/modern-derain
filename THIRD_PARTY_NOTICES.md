# Third-party provenance and notices

Project Python code is AGPL-3.0-only. External source, checkpoints and datasets retain their original terms.

## EfficientDeRain reference

Reference: https://github.com/tsingqguo/efficientderain at commit `465759da2428a4263eeee6ab464a0135fca2900c`.
Copyright (c) 2020 Qing Guo. MIT notice: `licenses/EfficientDeRain-MIT.txt`.
Unmodified files are test-only under `tests/reference/EfficientDeRain/465759da2428a4263eeee6ab464a0135fca2900c/`.
Production KPN and KernelConv are independently reconstructed with standard PyTorch operations, preserving module names, tensor alignment and operation order.

| Pinned reference files | Independent production implementation | Reason |
|---|---|---|
| `network.py` | `models/kpn.py`, `models/kernel.py` | Preserve v3 architecture, parameter names and RGB core ordering; remove unused attention, burst and alternative-loss paths. |
| `pytorch_ssim/__init__.py`, active loss in `trainer.py` | `training/losses.py` | Replace Variable/data APIs, correct device/dtype reuse, retain the original training SSIM and loss formula. |
| `dataset.py` | `data/images.py`, `data/augment.py`, `data/paired.py` | Retain RGB/range, common crop and ceil16 resize while implementing paired AlbumentationsX and contiguous tensors. |
| `utils.py` | `data/sources.py`, `data/images.py`, `evaluation/metrics.py` | Replace inactive/commented Rain1400 discovery with verified folder identities; retain recovery and image-wise PSNR semantics. |
| `trainer.py`, `train.py`, `train_rain1400.sh` | `config.py`, `training/loop.py` | Preserve active Adam/loss/schedule settings and add explicit state, data fingerprint and epoch-boundary resume checks. |
| `validation.py` | `evaluation/metrics.py`, `evaluation/quality.py` | Repair the upstream SSIM API name error and preserve float64 channel-wise evaluation under the modern API. |

File URLs, original byte sizes, Git blob IDs and SHA-256 values are in the oracle's `PROVENANCE.json`. Production modules never import these reference files. The oracle's obsolete optional imports are isolated in test-only loading and do not become package dependencies.

## Training SSIM

The vendored EfficientDeRain `pytorch_ssim/__init__.py` derives from Po-Hsun Su's pytorch-ssim: https://github.com/Po-Hsun-Su/pytorch-ssim.
The author's license declaration at commit `3add4532d3f633316cba235da1c69e90f0dfb952` is the literal `MIT`; it is preserved in `licenses/pytorch-ssim-LICENSE.txt` without adding an invented copyright notice.
Our loss removes deprecated Variable/data use and preserves Gaussian window construction, zero padding, population moments and L1−0.2×SSIM.

## Checkpoints and datasets

Official v3 Rain1400 weights are sourced from the archive linked by the pinned upstream README; details and verification are recorded in `weights/README.md`.
The extracted raw `.pth` is 109601091 bytes, SHA-256 `b8aaa9b5cd01177ca22a5344813923f91f298515be6382cf8654799cab5fa676`; the unchanged-tensor FP32 envelope is 109600519 bytes, SHA-256 `b7be892d6483e7e41b65f143ed1d6fc199cdaece8229f4a540e897a80abeeab6`. Original archive member: `models/v3_rain1400/v3_rain1400.pth` at http://www.xujuefei.com/models_effderain.zip (retrieved via HTTPS).
The verified envelope and checksum are hosted in this project's [versioned weights-v1 Release](https://github.com/Longcat2957/modern-derain/releases/tag/weights-v1); the default asset URL is https://github.com/Longcat2957/modern-derain/releases/download/weights-v1/efficientderain-rain1400-v3.pt. The original upstream `.pth` is not duplicated in the Release. The external checkpoint is not relicensed as project code.
The repository's MIT source license and our AGPL license do not establish separate checkpoint or dataset redistribution terms. No new license is assigned to external weights or datasets.

Rain1400 identity metadata comes from Xueyang Fu et al.'s [CVPR 2017 dataset page](https://xueyangfu.github.io/projects/cvpr2017.html) and its author-linked [`rainy_image_dataset.zip`](https://drive.google.com/file/d/10cu6MA4fQ2Dz16zfzyrQWDDhOWhRdhpq/view), observed on 2026-10-13. The archive is 898,292,342 bytes. HTTP byte ranges of its complete central directory and embedded readme establish training clean IDs 1–900 (12,600 rainy JPGs), testing IDs 901–1000 (1,400 rainy JPGs), and rain IDs 1–14 per clean image. The central-directory bytes SHA-256 is `63bec2f3ca9b5c0b090c25ec3193298c9356d924aff78e5b4b20785597f3fdab`. The archived uncompressed size/CRC32 records are preserved in `src/modern_derain/data/rain1400_identity.py`; ordered little-endian size/CRC32 record hashes and byte-range details are in `tests/fixtures/rain1400.json`. Ground-truth identity records SHA-256: `aff91ac1d902616044e057197f7f4f1729784e16222d782fa1adc6fac09eb83f`; rainy identity records SHA-256: `f0698cc1e6595523e17d05b797f475d65eff30528b8e55b7f99444c5b49146c8`. These are metadata hashes, not hashes of the full ZIP or image payloads. Runtime byte size/CRC32 checks identify the selected original images without decoding them; normal run snapshots separately use only path/size/mtime fingerprints. CRC32 is not a cryptographic integrity guarantee. The full dataset payload and quality benchmark were not acquired or evaluated.

The archive readme requests citations to Fu et al., *Removing Rain from Single Images via a Deep Detail Network* (CVPR 2017), and *Clearing the Skies: A Deep Network Architecture for Single-Image Rain Removal* (IEEE Transactions on Image Processing, 2017). It does not supply a separate dataset redistribution license; this project does not assign one or bundle the image dataset.

## AlbumentationsX

Training uses AlbumentationsX under its upstream AGPL-3.0/commercial licensing policy: https://albumentations.ai/docs/1-introduction/installation/.
It is an installed dependency, not vendored source. Dependencies retain their own licenses.

## Similar implementation

https://github.com/Longcat2957/modern-nafnet is a workflow reference only. Its restoration model, synthetic blur policy, and training defaults are not imported or copied.
