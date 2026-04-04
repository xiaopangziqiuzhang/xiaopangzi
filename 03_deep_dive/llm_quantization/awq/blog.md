# AWQ 深度解读：激活感知的权重量化

## 论文信息

- **标题**：AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
- **作者**：Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, Song Han
- **机构**：MIT, Tsinghua University
- **发表会议**：NeurIPS 2023
- **arXiv**：https://arxiv.org/abs/2306.00978
- **代码**：https://github.com/mit-han-lab/llm-awq

## 1. 核心创新点

AWQ是一种高效的低比特量化方法，主要创新点包括：

1. **保护重要权重**：发现只有约1%的权重对模型性能至关重要，量化时保护这些权重
2. **激活感知的通道缩放**：结合SmoothQuant的思想，进一步优化缩放策略
3. **硬件友好的设计**：专为GPU硬件设计，实现高效推理
4. **一键式量化**：使用简单，无需复杂调整
5. **性能优异**：在精度和速度上都优于GPTQ

## 2. 问题背景

（内容待完善）

## 3. 算法原理详解

（内容待完善）

## 4. 代码复现

（内容待完善）
