# xiaopangzi - 大模型量化研究仓库

> 🎯 目标：深入浅出地讲解大模型量化技术，让小白也能明白！

---

## 📚 仓库结构

```
xiaopangzi/
├── 00_getting_started/    🟢 入门级内容（小白请从这里开始！）
│   └── llm_quantization/
│       ├── beginners_guide.md       # 量化小白入门指南（生活例子讲明白）
│       └── visualize_quantization.py # 简单的量化可视化代码
│
├── 01_paper/               📄 论文研读
│   └── llm_quantization/
│       └── turbo_quant/            # TurboQuant 论文解读
│
├── 02_survey/              📖 综合调研
│   └── llm_quantization/
│       └── survey.md                 # 大语言模型量化技术调研（全面系统）
│
├── 03_deep_dive/           🔍 深度解读
│   └── llm_quantization/
│       ├── llm_int8/               # LLM.int8 原理+代码
│       ├── smoothquant/             # SmoothQuant 原理+代码
│       ├── gptq/                    # GPTQ 原理+代码
│       └── awq/                     # AWQ 原理+代码
│
└── 04_comparison/          📊 对比总结
    └── llm_quantization/
        └── method_comparison.md     # 量化方法对比与选择指南
```

---

## 🚀 学习路径

### 新手推荐顺序
1. **入门** → `00_getting_started/llm_quantization/beginners_guide.md`
2. **全貌** → `02_survey/llm_quantization/survey.md`
3. **深入** → `03_deep_dive/llm_quantization/` 选1-2个方法
4. **对比** → `04_comparison/llm_quantization/method_comparison.md`

### 快速选择方法
不知道选哪个量化方法？直接看 → `04_comparison/llm_quantization/method_comparison.md`

---

## 📝 内容特点

- ✅ **通俗易懂**：用生活例子讲解，避免天书
- ✅ **代码配套**：每个算法都有Python教学实现
- ✅ **体系完整**：从入门到精通，路径清晰
- ✅ **对比直观**：有表格、有决策树，帮你选择

---

## 💡 涵盖的量化方法

| 方法 | 年份 | 特点 | 推荐指数 |
|------|------|------|---------|
| LLM.int8 | 2022 | 零损失，INT8 | ⭐⭐⭐⭐ |
| SmoothQuant | 2022 | 简单快速，INT8 | ⭐⭐⭐⭐⭐ |
| GPTQ | 2023 | 精度高，INT4 | ⭐⭐⭐⭐ |
| AWQ | 2023 | 精度高+速度快，INT4 | ⭐⭐⭐⭐⭐ |

---

## 🤝 贡献

欢迎提Issue和PR！如果你发现有不清楚的地方，或者想补充内容，尽管来！

---

## 📄 许可证

MIT License

---

## 🙏 致谢

感谢所有量化领域的研究者们！这个仓库站在了巨人的肩膀上。

---

*祝大家学习愉快！有问题欢迎交流！* 🎉

