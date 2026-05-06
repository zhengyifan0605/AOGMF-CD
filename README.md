# AOGMF-CD

AOGMF-CD is a remote-sensing change detection project. The main model entry is `AOGMF` in `network/AOGMF.py`, with training and testing scripts provided by `train_AOGMF.py` and `test.py`.

## Reviewer Quick Start

Use Git LFS when cloning the repository, because the default model checkpoint is about 159M.

```bash
git lfs install
git clone <repo-url>
cd AOGMF-CD
git lfs pull
```

For Git LFS repositories, use `git clone` rather than a source-code ZIP download so that the checkpoint is available as a real `.pth` file instead of an LFS pointer file.

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate cgnet
```

Verify that the checkpoint was downloaded correctly:

```bash
sha256sum weights/AOGMF_WHU_best_iou.pth
```

Expected SHA256:

```text
f8fd8068f3c4dfb3035d66b4da8e355ba8114a00df402cee4b9f2b1b8301b031
```

Run the included sample test:

```bash
python test.py \
  --gpu_id 0 \
  --data_name WHU \
  --model_name AOGMF \
  --batchsize 1 \
  --load weights/AOGMF_WHU_best_iou.pth \
  --test_root sample_data/WHU/test/ \
  --save_path test_result_smoke/
```

The expected behavior is that the script loads `weights/AOGMF_WHU_best_iou.pth`, runs one test image pair from `sample_data/WHU/test/`, and writes a prediction image under `test_result_smoke/WHU/AOGMF_APM+MBDCB+PFF+CGA/`.

## Environment

The environment used for verification was created on a GPU machine with NVIDIA RTX 4090 and CUDA available:

```bash
conda env create -f environment.yml
conda activate cgnet
```

The scripts call `.cuda()` directly, so an NVIDIA GPU with a working CUDA driver is required for training and testing unless the code is modified for CPU execution.

If the environment already exists, update it with:

```bash
conda env update -n cgnet -f environment.yml
```

Verified package versions in the local `cgnet` environment:

- Python 3.10
- PyTorch 2.4.0
- torchvision 0.19.0
- thop 0.1.1.post2209072238

The model code uses `torchvision.models.vgg16_bn(pretrained=True)`. On the first model construction, torchvision may download the VGG16-BN ImageNet pretrained weight `vgg16_bn-6c64b313.pth` into the local Torch cache.

## Data Layout

Training, validation, and test directories should use this layout:

```text
dataset/WHU-CD256-HANet/
  train/
    A/
    B/
    label/
  val/
    A/
    B/
    label/
  test/
    A/
    B/
    label/
```

A minimal test sample is included under:

```text
sample_data/WHU/test/
  A/
    test256_0_309_4.png
  B/
    test256_0_309_4.png
  label/
    test256_0_309_4.png
```

The sample is intended for command and I/O verification only. Metrics from a single sample may be `nan` for classes that are absent in the label or prediction.

## Weights

Put the default test weight at:

```text
weights/AOGMF_WHU_best_iou.pth
```

The local verified file has SHA256:

```text
f8fd8068f3c4dfb3035d66b4da8e355ba8114a00df402cee4b9f2b1b8301b031
```

The file is about 159M. A normal GitHub Git push rejects files over 100M, so publish this weight through Git LFS or a GitHub Release asset while keeping the same local path.
This repository is prepared for the Git LFS path by tracking `weights/AOGMF_WHU_best_iou.pth` in `.gitattributes`.

## Train

Example WHU training command:

```bash
python train_AOGMF.py \
  --gpu_id 0 \
  --data_name WHU \
  --model_name AOGMF \
  --batchsize 8 \
  --epoch 50 \
  --train_root dataset/WHU-CD256-HANet/train/ \
  --val_root dataset/WHU-CD256-HANet/val/
```

The script writes checkpoints under `output_ablation/WHU/AOGMF_APM+MBDCB+PFF+CGA*` by default.

## Test

Smoke test with the included sample:

```bash
python test.py \
  --gpu_id 0 \
  --data_name WHU \
  --model_name AOGMF \
  --batchsize 1 \
  --load weights/AOGMF_WHU_best_iou.pth \
  --test_root sample_data/WHU/test/ \
  --save_path test_result_smoke/
```

Full WHU test:

```bash
python test.py \
  --gpu_id 0 \
  --data_name WHU \
  --model_name AOGMF \
  --batchsize 1 \
  --load weights/AOGMF_WHU_best_iou.pth \
  --test_root dataset/WHU-CD256-HANet/test/ \
  --save_path test_result_ablation/
```

In restricted environments where matplotlib or font caches are not writable, run the same command with temporary cache directories:

```bash
conda run -n cgnet env MPLCONFIGDIR=/tmp/aogmf-mpl XDG_CACHE_HOME=/tmp/aogmf-cache python test.py \
  --gpu_id 0 \
  --data_name WHU \
  --model_name AOGMF \
  --batchsize 1 \
  --load weights/AOGMF_WHU_best_iou.pth \
  --test_root sample_data/WHU/test/ \
  --save_path test_result_smoke/
```

## Verification

The following checks were run successfully in the local environment:

```bash
python -m py_compile test.py train_AOGMF.py network/AOGMF.py
conda run -n cgnet python train_AOGMF.py --help
conda run -n cgnet python test.py --help
conda run -n cgnet env MPLCONFIGDIR=/tmp/aogmf-mpl XDG_CACHE_HOME=/tmp/aogmf-cache python test.py --gpu_id 0 --data_name WHU --model_name AOGMF --batchsize 1 --load weights/AOGMF_WHU_best_iou.pth --test_root sample_data/WHU/test/ --save_path test_result_smoke/
```
