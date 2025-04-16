# SurvTreeSHAP

**SurvTreeSHAP** is a Python package for interpreting tree-based survival models at scale. It extends SurvSHAP(t) explanations addressing the computational bottlenecks with the TreeSHAP algorithm. It takes advantage of the tree structure and provide fast and scalable survival prediction explanations in high-dimensional settings.

## 🔧 Installation

```bash
pip install survtreeshap

```

## 📦 Supported Models

- `scikit-survival` Random Survival Forests (`RandomSurvivalForest`)
- `xgboost` or `lightgbm` survival models with appropriate loss functions

## 💡 Example Usage

```python
from survtreeshap import SurvSHAPExplainer
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

## 📚 Citation

If you use this package in your research, please cite:

@article{yourname2025scalablesurvshap,
  title={Scalable SHAP-based Interpretation for Tree-based Survival Models},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:2504.xxxxx},
  year={2025}
}

## 📄 License

MIT License
