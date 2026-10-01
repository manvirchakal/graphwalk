# Router analysis: predicting walk success before the walk

Label: walk F1 >= 0.5. AUROC from 5-fold cross-validated logistic regression (pre-walk features; + confidence) or confidence alone; 95% bootstrap CIs.
Features: words, entities, superlative, temporal, conjunction, counting, wh_who, wh_what, wh_where, wh_when, wh_which, wh_how, log_degree, relations_1hop, log_relations_2hop.

| data | n | walk right | pre-walk AUROC | confidence AUROC | pre+conf AUROC |
|---|---|---|---|---|---|
| metaqa | 1760 | 0.94 | 0.68 [0.62, 0.73] | 0.96 [0.94, 0.98] | 0.97 [0.96, 0.98] |
|  | top single features: words 0.35, relations_1hop 0.38, wh_what 0.59, log_degree 0.44 | | | | |
| webqsp | 50 | 0.54 | 0.55 [0.38, 0.72] | 0.89 [0.78, 0.97] | 0.86 [0.75, 0.95] |
|  | top single features: words 0.37, temporal 0.41, relations_1hop 0.43, log_relations_2hop 0.43 | | | | |
| cwq | 50 | 0.32 | 0.41 [0.23, 0.60] | 0.70 [0.52, 0.87] | 0.50 [0.32, 0.68] |
|  | top single features: wh_what 0.59, wh_which 0.42, wh_where 0.55, log_relations_2hop 0.54 | | | | |
| webqsp-global | 100 | 0.53 | 0.62 [0.50, 0.73] | 0.83 [0.74, 0.91] | 0.76 [0.67, 0.85] |
|  | top single features: wh_what 0.62, log_relations_2hop 0.40, log_degree 0.41, temporal 0.42 | | | | |
| freebase (pooled) | 200 | 0.48 | 0.63 [0.55, 0.70] | 0.83 [0.77, 0.88] | 0.78 [0.72, 0.84] |
|  | top single features: words 0.38, wh_what 0.61, temporal 0.43, log_degree 0.44 | | | | |
| all (pooled) | 1960 | 0.90 | 0.81 [0.77, 0.84] | 0.94 [0.92, 0.96] | 0.96 [0.95, 0.97] |
|  | top single features: relations_1hop 0.22, log_degree 0.25, log_relations_2hop 0.27, entities 0.46 | | | | |
