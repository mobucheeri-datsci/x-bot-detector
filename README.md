# Capstone Project: X (formerly Twitter) Bot Detection
## A Google Chrome Browser Extension for Real-Time Bot Identification of User Profiles on X

Student: Mohamed Bucheeri  
Programme: General Assembly Data Science PT2 (BIBF Bahrain)  
Date: April 11 2026

### Introduction
This project trains a machine learning model to detect bot accounts on X and ships it as a Google Chrome extension that scores any profile in real time. I trained three classifiers (BiGRU-LSTM, CNN-BiLSTM, XGBoost) on 37,438 labelled accounts and kept the best one. A FastAPI backend serves the model and the extension calls it. Every score comes with a written breakdown of which features pushed it up or down.

### Problem Statement
Automated accounts make up a large and growing share of activity on social media. The 2024 Imperva Bad Bot Report found that almost half of all internet traffic in 2023 was automated, with bad bots responsible for 32% of total traffic (Imperva, 2024). On X, automated accounts amplify low-credibility content (Shao et al., 2018), distort political discourse during elections (Bessi and Ferrara, 2016; Howard, Woolley and Calo, 2018), and increase exposure to negative material (Stella, Ferrara and De Domenico, 2018). As an active user on X, I find it difficult to determine what's real or manufactured (Ferrara et al., 2016; Cresci, 2020). The bot detection tools that already exist tend to need academic API access, or live on a separate website, or require a paid subscription. None of them run inside the browser where the user already is. That is the gap this project fills.

### Project Aim
To build a machine learning system that detects bot accounts on X from only the profile metadata a browser extension can see, and to ship it as a working tool anyone can install.

### Project Objectives
1. Find and preprocess a public labelled dataset of X accounts.
2. Engineer features that capture how bots differ from real users.
3. Train and compare three classifier families (RNN, CNN, gradient-boosted trees).
4. Address class imbalance through weighted loss and tune the decision threshold per model.
5. Select the best model by F1 on a test set and prepare it to be deployed.
6. Handle the model using a FastAPI backend with prediction endpoints.
7. Build a Chrome extension that scrapes X profiles and displays bot scores in real time.
8. Show which features pushed each score up or down, written as readable descriptions, across the popup, the inline panel, and the thread hover cards.
9. Compare results with an external model.

### Executive Summary
The XGBoost model trained on 37 engineered numeric features reached F1 = 0.8076, ROC-AUC = 0.9329, and accuracy = 0.8725 on a held-out test set of 5,616 users. BiGRU-LSTM and CNN-BiLSTM came in 1.6 and 5.7 F1 points behind. The top features by gain importance were `verified` (0.20), `followers_count` (0.11), `is_established_account` (0.10), and `log_followers_count` (0.07). Gradient-boosted trees beating deep learning on tabular features matches what the literature reports (Shwartz-Ziv and Armon, 2022; Grinsztajn, Oyallon and Varoquaux, 2022). The Chrome extension shows the same per-feature breakdown wherever it scores something. That means the inline panel on profile pages, the toolbar popup, and the hover card on each reply dot in a tweet thread. It also runs unsupervised clustering on visible reply features to flag groups of similar accounts as possible coordinated inauthentic behaviour. A custom benchmark of 50 verified organisations confirmed the model classifies all 50 as HUMAN once it has complete metadata. A comparison with the MGTAB model (Liu et al., 2023) reached F1 = 0.8364, close to MGTAB's published Random Forest baseline.

### Audience
Everyday social media users who want a quick way to check suspicious accounts. Researchers, journalists, and fact-checkers who need a fast screening tool. Data science students who want a worked end-to-end ML pipeline.

### Data Sources
`twitter_human_bots.csv` (airt-ml, 2023). 37,438 X accounts with profile metadata (followers, friends, statuses, favourites, account age, verified status, default profile flags, bio, screen name, location) and a binary `account_type` label. Hosted on Hugging Face under CC BY-SA 3.0. Downloaded automatically on first run via the `datasets` library.

### Methodology
1. Download dataset from Hugging Face and cache to disk.
2. Clean raw fields, fill missing values, cast types.
3. Engineer 37 numeric features grouped into 8 categories.
4. Random 70/15/15 train/val/test split with a fixed random seed.
5. Normalise numeric features with a standard scaler fitted on the training split only.
6. Build a 50,000 word vocabulary from the training corpus for the GloVe-based text models.
7. Train BiGRU-LSTM, CNN-BiLSTM, and XGBoost using class-weighted loss to handle the 2:1 imbalance.
8. Tune the decision threshold per model on the validation set.
9. Evaluate on the held-out test set using F1, accuracy, precision, recall, ROC-AUC, and PR-AUC.
10. Save the best model by F1 and wire it into a FastAPI backend with `/predict` and `/predict_batch` endpoints.
11. Build the Chrome extension with single-profile scoring, multi-context thread analysis, and CIB clustering.
12. Validate the model's results using an external model, the MTGAB.

### File Directory
```
twitter-bot-detector/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/twitter_human_bots.csv
│   └── processed/processed.pt
├── models/checkpoints/      (gitignored, regenerated by src/train.py)
├── src/
│   ├── config.py            paths and hyperparameters
│   ├── data.py              download and feature engineering
│   ├── models.py            BiGRU-LSTM and CNN-BiLSTM
│   ├── train.py             training loop, threshold tuning, save best
│   ├── embed_bios.py        Sentence-BERT bio embeddings
│   ├── orgs_eval.py         50-organisation benchmark
│   └── mgtab_eval.py        MGTAB architecture-transfer
├── api/app.py               FastAPI backend
├── extension/
│   ├── manifest.json
│   ├── content/             DOM scraping, profile panel, thread analysis
│   ├── popup/               toolbar popup
│   ├── background/          service worker
│   └── icons/
└── notebooks/analysis.ipynb
```

### Data Dictionary

#### Raw fields from `twitter_human_bots.csv`
| Column | Type | Description |
|---|---|---|
| id | int64 | Unique X user ID |
| screen_name | object | X handle |
| description | object | Profile bio text |
| location | object | Self-reported location |
| created_at | datetime | Account creation timestamp |
| followers_count | int64 | Number of followers |
| friends_count | int64 | Number of accounts followed |
| statuses_count | int64 | Total tweets posted |
| favourites_count | int64 | Total tweets liked |
| verified | bool | Verified status |
| default_profile | bool | Default profile theme |
| default_profile_image | bool | Default profile image |
| account_age_days | int64 | Days since creation |
| average_tweets_per_day | float64 | Mean tweets per day |
| account_type | object | Target label: `human` or `bot` |

#### Engineered features (37 total, 8 categories)
| Category | Count |
|---|---|
| Raw counts | 6 |
| Log transforms | 6 |
| Behavioural ratios | 4 |
| Profile completeness | 8 |
| Screen name patterns | 3 |
| Activity anomalies | 2 |
| Bio content signals | 4 |
| Account-type signals | 4 |

The full ordered list is in `src/config.py` as `numeric_features`. All features are normalised with a z-score scaler fitted on training data only to avoid data leakage.

### Model Architectures
BiGRU-LSTM with gated fusion: Bidirectional GRU followed by an LSTM on word embeddings of the user's bio. Text branch combined with a `NumericNet` over the 37 numeric features through a learned sigmoid gate that decides per-sample which branch to weigh more.

CNN-BiLSTM with gated fusion: Multi-kernel 1D CNN (kernel sizes 3, 4, 5) plus max-over-time pooling and a bidirectional LSTM. Same `NumericNet` and gated fusion.

XGBoost: Gradient-boosted trees on the 37 engineered features. `scale_pos_weight = 2.03` for the 2:1 imbalance.

All three use class-weighted loss, early stopping on validation F1, and a per-model decision threshold tuned on the validation set.

### Results
| Model | F1 | Accuracy | Precision | Recall | ROC-AUC | PR-AUC | Threshold | Train Time |
|---|---|---|---|---|---|---|---|---|
| XGBoost | 0.8076 | 0.8725 | 0.8336 | 0.7832 | 0.9329 | 0.9031 | 0.580 | 1 s |
| BiGRU-LSTM | 0.7914 | 0.8549 | 0.7777 | 0.8056 | 0.9220 | 0.8872 | 0.520 | 829 s |
| CNN-BiLSTM | 0.7503 | 0.8292 | 0.7497 | 0.7509 | 0.9001 | 0.8486 | 0.570 | 49 s |

XGBoost has the highest F1 and accuracy. Training time differs by three orders of magnitude (1 s for XGBoost vs 829 s for BiGRU-LSTM). XGBoost weights `verified` highest at 0.20 followed by raw `followers_count` at 0.11 and the engineered `is_established_account` flag at 0.10. A Random Forest baseline puts log-transformed counts at the top of the importance ranking with raw counts close behind. Full charts, ROC curves, and confusion matrices are in `notebooks/analysis.ipynb`.

### Benchmark on 50 Verified Organisations
A benchmark of 50 verified organisations (15 news, 10 tech, 10 retail, 8 sports, 7 NGOs and government) tests how the model handles legitimate organisation accounts. The model classifies all 50 as HUMAN on its own with no organisation scoring above 50/100. The news-organisation override only changes the score for 2 of 50 accounts (NFL and F1). 

### External Validation on MGTAB Model
MGTAB (Liu et al., 2023, arXiv:2301.01123) is a published bot detection dataset with 10,199 expert-annotated users and a multi-relational graph. It ships only as preprocessed PyTorch tensors with 788 pre-extracted features per user. We trained the same XGBoost architecture on its features to validate the results.

| Model | F1 on MGTAB |
|---|---|
| Random Forest (Liu et al., 2023) | ~0.84 |
| GCN (Liu et al., 2023) | ~0.86 |
| RGT (Liu et al., 2023) | ~0.89 |
| XGBoost | 0.8364 |

The XGBoost matches MGTAB's published Random Forest baseline. Graph-based models reach a few points higher because MGTAB's contribution is its multi-relational graph, which feature-only models cannot use. Implementation in `src/mgtab_eval.py`.

### Reproducing the Results
Trained checkpoints, the preprocessed dataset, the SBERT embedding cache, and MGTAB are gitignored. The notebook is committed with cell outputs preserved, so it renders on GitHub without re-running. To reproduce from a fresh clone:

```bash
git clone https://git.generalassemb.ly/mobucheeri/twitter-bot-detector.git
cd twitter-bot-detector
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/data.py
python src/embed_bios.py
python src/train.py
python src/orgs_eval.py

pip install gdown
mkdir -p data/raw/mgtab && cd data/raw/mgtab
gdown 1gbWNOoU1JB8RrTu2a5j9KMNVa9wX72Fe -O mgtab.zip && unzip mgtab.zip
cd ../../..
python src/mgtab_eval.py

uvicorn api.app:app --reload --port 8000
```

Then in Chrome: `chrome://extensions` -> enable Developer mode -> Load unpacked -> select `extension/`. Navigate to any X profile or tweet detail page to see the inline panel, popup, and thread analysis.

### Conclusion
The deployed XGBoost model reaches F1 = 0.8076, ROC-AUC = 0.9329, and accuracy = 0.8725 on the test set of 5,616 users. BiGRU-LSTM lands 1.6 F1 points behind and CNN-BiLSTM lands 5.7 points behind, even though both neural models have around 10 million parameters and the XGBoost does not. The four highest-importance features by gain are `verified` (0.20), `followers_count` (0.11), `is_established_account` (0.10), and `log_followers_count` (0.07). The Sentence-BERT bio embedding fusion experiment in section 6 regresses by 0.0025 F1 against the baseline. 19% of bios in the dataset are empty and the median bio length is 53 characters, so most are shorter than what SBERT was trained on. This is consistent with Shwartz-Ziv and Armon (2022).

The 50-organisation benchmark in classifies all 50 verified organisations as HUMAN when given complete metadata, and the news-organisation override only changes the score for 2 of them (NFL and F1). The live extension errors on Al Jazeera (84/100) and CNN (92/100) trace back to incomplete metadata in the DOM scraping path of `content.js`, not to the model. Improving the scraper to read more profile fields from the hover cards is the highest-impact follow-up for the deployed product. The comparison with MGTAB (Liu et al., 2023) reached F1 = 0.8364 on the 788 pre-computed features, around the level of MGTAB's published Random Forest baseline of 0.84. The graph-based models on the same dataset reach 0.86 to 0.89, and the gap there is the social graph that a browser extension cannot see at inference. Within the profile-only constraint, the deployed XGBoost plus the per-prediction contributions give an everyday user a way to screen suspicious accounts without needing API access or paid subscription.

### Recommendations
1. Use the extension as a general screening for bot detection when using it on the site. Manual review should still be carried out regularly.
2. Prefer gradient-boosted trees over deep learning when the features are tabular.
3. Engineer ratios and log transforms, not just raw counts. Most of the gains came from features, not architecture.
4. Tune the decision threshold per model. Default 0.5 leaves performance on the table when classes are imbalanced.

### Areas for Further Research
1. Tweet content features. The current dataset only has bios, not tweet histories.
2. Graph neural networks. On TwiBot-22 (Feng et al., 2022) and MGTAB they sit a few F1 points above feature-only models, but the graph data is not available to a browser extension at inference time.
3. Cross-platform generalisation to Instagram, Threads, Reddit, or YouTube.
4. Modern LLM-driven bots. The current dataset predates ChatGPT-style accounts.
5. Time-sensing behavioural signals from visible tweet timestamps.

### Limitations
The dataset is from 2018-2020 and contains no modern LLM-driven bot accounts. The deployed model only sees profile-level features, which means it cannot use the social graph or full tweet history. A Sentence-BERT bio embedding fusion experiment produced a small F1 regression (-0.0025), so the deployed model stays at the 37-feature baseline. The live extension misclassifications on Al Jazeera and CNN trace to incomplete DOM scraping in `content.js`, not to the model itself. Therefore, improving the scraper to recover follower count and account age from hover cards would be an inpactful change to make post-deployment.

### References
airt-ml (2023). *Twitter Human Bots Dataset*. Hugging Face. Available at: https://huggingface.co/datasets/airt-ml/twitter-human-bots [Accessed April 2026].

Bessi, A. and Ferrara, E. (2016). 'Social bots distort the 2016 U.S. presidential election online discussion'. *First Monday*, 21(11).

Chen, T. and Guestrin, C. (2016). 'XGBoost: A scalable tree boosting system'. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785 to 794.

Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H. and Bengio, Y. (2014). 'Learning phrase representations using RNN encoder-decoder for statistical machine translation'. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 1724 to 1734.

Cresci, S. (2020). 'A decade of social bot detection'. *Communications of the ACM*, 63(10), pp. 72 to 83.

Feng, S., Tan, Z., Wan, H., Wang, N., Chen, Z., Zhang, B., Zheng, Q., Zhang, W., Lei, Z., Yang, S., Feng, X. and Luo, M. (2022). 'TwiBot-22: Towards graph-based Twitter bot detection'. *Advances in Neural Information Processing Systems*, 35, pp. 35254 to 35269.

Ferrara, E., Varol, O., Davis, C., Menczer, F. and Flammini, A. (2016). 'The rise of social bots'. *Communications of the ACM*, 59(7), pp. 96 to 104.

Grinsztajn, L., Oyallon, E. and Varoquaux, G. (2022). 'Why do tree-based models still outperform deep learning on typical tabular data?'. *Advances in Neural Information Processing Systems*, 35, pp. 507 to 520.

Hochreiter, S. and Schmidhuber, J. (1997). 'Long short-term memory'. *Neural Computation*, 9(8), pp. 1735 to 1780.

Howard, P.N., Woolley, S. and Calo, R. (2018). 'Algorithms, bots, and political communications in the US 2016 election'. *Journal of Information Technology and Politics*, 15(2), pp. 81 to 93.

Imperva (2024). *2024 Bad Bot Report*. Available at: https://www.imperva.com/resources/resource-library/reports/2024-bad-bot-report/ [Accessed April 2026].

Kim, Y. (2014). 'Convolutional neural networks for sentence classification'. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 1746 to 1751.

Liu, C., Wang, T., Zhou, X., Liu, J., Yu, X., Zhao, Y. and Lin, X. (2023). 'MGTAB: A multi-relational graph-based Twitter account detection benchmark'. *arXiv preprint*, arXiv:2301.01123.

Pennington, J., Socher, R. and Manning, C. (2014). 'GloVe: Global vectors for word representation'. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 1532 to 1543.

Reimers, N. and Gurevych, I. (2019). 'Sentence-BERT: Sentence embeddings using Siamese BERT-networks'. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 3982 to 3992.

Shao, C., Ciampaglia, G.L., Varol, O., Yang, K.C., Flammini, A. and Menczer, F. (2018). 'The spread of low-credibility content by social bots'. *Nature Communications*, 9(1), p. 4787.

Shwartz-Ziv, R. and Armon, A. (2022). 'Tabular data: Deep learning is not all you need'. *Information Fusion*, 81, pp. 84 to 90.

Stella, M., Ferrara, E. and De Domenico, M. (2018). 'Bots increase exposure to negative and inflammatory content in online social systems'. *Proceedings of the National Academy of Sciences*, 115(49), pp. 12435 to 12440.
