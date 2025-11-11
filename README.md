# Swin Transformer 注意力热力图可视化

本仓库提供了一个轻量级推理流程，用于从经过微调的 **Swin Transformer** 模型中生成任务相关的注意力热力图（Attention Heatmaps）。  
脚本会从 `ckpt/` 目录加载模型权重，对输入图像执行多任务推理，并保存各下游任务的注意力区域，可视化模型在不同任务中的关注位置。

---

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

生成注意力热力图:

```bash
python scripts/generate_attention_heatmaps.py \
    --ckpt ckpt/ \
    --image data/2008_000008.jpg \
    --output outputs/
```

脚本默认会运行以下四个任务:

1. Semantic segmentation
2. Human part detection
3. Surface normals estimation
4. Saliency distillation

对于每个检查点（checkpoint），脚本都会在指定的输出路径下创建一个独立的子目录（例如 `outputs/my_checkpoint/`），并在其中保存四个任务对应的 **`.png` 叠加图像** 和原始的 **`.npy` 热力图文件**。


## Notes

* 预训练或微调模型应基于 timm 库中的 Swin Transformer 架构。
* 为了提取注意力矩阵，本代码默认禁用 fused attention kernel 加速。
* 若需要添加新的任务，可修改scripts/generate_attention_heatmaps.py 文件中的 default_task_specs() 函数。