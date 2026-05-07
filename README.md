# AOGMF-CD

AOGMF-CD 是一个遥感变化检测项目。主模型入口为 `network/AOGMF.py` 中的 `AOGMF`，训练与测试脚本分别为 `train_AOGMF.py` 和 `test.py`。

## 验收快速开始

克隆仓库或下载源码 ZIP：

```bash
git clone https://github.com/zhengyifan0605/AOGMF-CD
cd AOGMF-CD
```

从以下地址下载 Release 资产 `AOGMF_WHU_best_iou.pth`：

```text
https://github.com/zhengyifan0605/AOGMF-CD/releases
```

然后将其放到：

```text
weights/AOGMF_WHU_best_iou.pth
```

创建并激活环境：

```bash
conda env create -f environment.yml
conda activate cgnet
```

验证权重文件是否下载正确：

```bash
sha256sum weights/AOGMF_WHU_best_iou.pth
```

预期的 SHA256 为：

```text
f8fd8068f3c4dfb3035d66b4da8e355ba8114a00df402cee4b9f2b1b8301b031
```

运行仓库附带的样例测试：

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

预期行为是：脚本会加载 `weights/AOGMF_WHU_best_iou.pth`，对 `sample_data/WHU/test/` 中的一组测试图像进行推理，并将预测结果写入 `test_result_smoke/WHU/AOGMF_APM+MBDCB+PFF+CGA/`。

## 环境配置

本项目验证时使用的是带 NVIDIA RTX 4090 且 CUDA 可用的 GPU 机器，环境创建方式如下：

```bash
conda env create -f environment.yml
conda activate cgnet
```

脚本中直接调用了 `.cuda()`，因此训练和测试默认都需要带有可用 CUDA 驱动的 NVIDIA GPU，除非你自行修改代码以支持 CPU 执行。


本地 `cgnet` 环境中验证通过的关键包版本如下：

- Python 3.10
- PyTorch 2.4.0
- torchvision 0.19.0
- thop 0.1.1.post2209072238

模型代码中使用了 `torchvision.models.vgg16_bn(pretrained=True)`。首次构建模型时，`torchvision` 可能会将 VGG16-BN 的 ImageNet 预训练权重 `vgg16_bn-6c64b313.pth` 下载到本地 Torch 缓存目录。

## 数据目录结构

训练、验证和测试数据建议使用如下目录结构：

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

仓库中附带了一个最小测试样例，目录如下：

```text
sample_data/WHU/test/
  A/
    test256_0_9_3.png
  B/
    test256_0_9_3.png
  label/
    test256_0_9_3.png
```

该样例仅用于命令和输入输出流程验证。由于只有单个样本，当标签或预测中缺失某一类别时，评估指标中可能出现 `nan`。

## 权重说明

从 GitHub Releases 页面下载默认的 WHU 测试权重，并放到：

```text
weights/AOGMF_WHU_best_iou.pth
```

Release 页面：

```text
https://github.com/zhengyifan0605/AOGMF-CD/releases
```

下载资产 `AOGMF_WHU_best_iou.pth` 后，请保持文件名不变，并直接放置在 `weights/` 目录下。

本地验证通过的权重文件 SHA256 为：

```text
f8fd8068f3c4dfb3035d66b4da8e355ba8114a00df402cee4b9f2b1b8301b031
```

该文件大小约为 159M，因此通过 GitHub Releases 分发，而不是直接跟踪在 Git 仓库中。

## 训练

WHU 数据集训练命令示例：

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

脚本默认会将 checkpoint 保存到 `output_ablation/WHU/AOGMF_APM+MBDCB+PFF+CGA*`。

## 测试

使用仓库内置样例进行 smoke test：

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

对 WHU 完整测试集进行测试：

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

如果运行环境中 `matplotlib` 或字体缓存目录不可写，可使用临时缓存目录运行相同命令：

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

## 验证记录

以下检查已在本地环境中验证通过：

```bash
python -m py_compile test.py train_AOGMF.py network/AOGMF.py
conda run -n cgnet python train_AOGMF.py --help
conda run -n cgnet python test.py --help
conda run -n cgnet env MPLCONFIGDIR=/tmp/aogmf-mpl XDG_CACHE_HOME=/tmp/aogmf-cache python test.py --gpu_id 0 --data_name WHU --model_name AOGMF --batchsize 1 --load weights/AOGMF_WHU_best_iou.pth --test_root sample_data/WHU/test/ --save_path test_result_smoke/
```
