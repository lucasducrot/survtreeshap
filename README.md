# SurvTreeSHAP

**SurvTreeSHAP** is a Python package for interpreting tree-based survival models at scale. It extends SurvSHAP(t) explanations addressing the computational bottlenecks with the TreeSHAP algorithm. It takes advantage of the tree structure and provide fast and scalable survival prediction explanations in high-dimensional settings.


## 📦 Supported Models

- `scikit-survival` Random Survival Forests (`RandomSurvivalForest`)
- `xgboost` or `lightgbm` survival models with appropriate loss functions

## 💡 Example Usage

See examples in jupyter notebooks for simulations and real dataset GBSG

## 📚 Citation

If you use this package in your research, please cite:

@inproceedings{ducrot:hal-05108033,
  TITLE = {{SurvTreeSHAP(t) : scalable explanation method for tree-based survival models}},
  AUTHOR = {Ducrot, Lucas and Fevry, Gilles and Dano, Clement and Texier, Raphael and Murris, Juliette and Katsahian, Sandrine},
  URL = {https://hal.science/hal-05108033},
  BOOKTITLE = {{Workshop on Explainable Artificial Intelligence (XAI), IJCAI 2025}},
  ADDRESS = {Montreal, Canada},
  YEAR = {2025},
  MONTH = Aug,
  KEYWORDS = {Medical applications ; Machine learning ; TreeSHAP ; SurvSHAP ; SHAP ; Explainability ; Interpretability ; Tree-based model ; Survival analysis},
  PDF = {https://hal.science/hal-05108033v1/file/SurvTreeSHAP___scalable_explanation_method_for_tree_based_survival_models.pdf},
  HAL_ID = {hal-05108033},
  HAL_VERSION = {v1},
}
## 📄 License

GNU GPL v3
