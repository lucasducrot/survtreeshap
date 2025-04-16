# ScalableSurvSHAP

**ScalableSurvSHAP** is a Python package for interpreting tree-based survival models at scale. It extends SHAP-based explanations to survival analysis while addressing the computational bottlenecks of existing approaches like SurvSHAP. This tool is specifically designed for Random Survival Forests and Gradient Boosted Survival Trees, enabling efficient and interpretable survival prediction explanations in high-dimensional settings.

## 🚀 Features

- ⚡ **Scalable SHAP explanations** for survival models trained on hundreds to thousands of covariates.
- 🌳 **Model-specific support** for Random Survival Forests (RSF) and Gradient Boosted Survival Trees.
- 📈 **Time-dependent feature attributions** for survival functions and cumulative hazard predictions.
- 🧠 **Faithful to TreeSHAP and SurvSHAP** logic, with optimized implementation for speed and memory efficiency.
- 🧪 Easy integration into survival model pipelines for clinical or biomedical applications.

## 🔧 Installation

```bash
pip install scalablesurvshap

