# SurvTreeSHAP

**SurvTreeSHAP** is a Python package for interpreting tree-based survival models at scale. It extends SHAP-based explanations to survival analysis while addressing the computational bottlenecks of existing approaches like Kernel SurvSHAP. This tool is an extension of TreeSHAP and specifically designed for Random Survival Forests and Gradient Boosted Survival Trees, enabling efficient and interpretable survival prediction explanations in high-dimensional settings.

## 🚀 Features

- 🌳 **Tree-based survival models**
- ⚡ **Scalable SHAP explanations**
- 📈 **Time-dependent feature attributions**

## 🔧 Installation

```bash
pip install survtreeshap

```

## 📦 Supported Models

- `scikit-survival` Random Survival Forests (`RandomSurvivalForest`)
- `xgboost` or `lightgbm` survival models with appropriate loss functions
- Custom survival tree models with SHAP-compatible APIs

## 🧬 Example Usage

```python
from scalablesurvshap import SurvSHAPExplainer
from sksurv.ensemble import RandomSurvivalForest
from sksurv.datasets import load_whas500
from sklearn.model_selection import train_test_split

# Load example data
X, y = load_whas500()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a survival model
model = RandomSurvivalForest(n_estimators=100).fit(X_train, y_train)

# Create an explainer
explainer = SurvSHAPExplainer(model, X_train)

# Compute SHAP values for one or more individuals
shap_values = explainer.explain(X_test.iloc[:10], time_horizon=365)

# Plot results
explainer.plot_summary(shap_values)
```

## 🧪 Citation

If you use this package in your research, please cite:

@article{yourname2025scalablesurvshap,
  title={Scalable SHAP-based Interpretation for Tree-based Survival Models},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:2504.xxxxx},
  year={2025}
}

## 📄 License

MIT License
