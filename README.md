# Capstone Project: X (formerly Twitter) Bot Detection
## A Google Chrome Browser Extension for Real-Time Bot Identification of User Profiles on X

Student: Mohamed Bucheeri
Programme: General Assembly Data Science PT2 (BIBF Bahrain)
Date: April 2026

### Introduction
This project trains a machine learning model to detect bot accounts on X and deploys it as a Google Chrome browser extension that makes a decision on any profile in real time. The pipeline covers data preprocessing, feature engineering, model training, comparison, deployment via a FastAPI backend, and integration with the extension. Three classifiers are compared on 37,438 labelled accounts: BiGRU-LSTM, CNN-BiLSTM, and XGBoost. The best model is selected automatically, served from the API, and consumed by the extension.

### Problem Statement
Automated accounts make up a large and growing share of activity on social media. The 2024 Imperva Bad Bot Report found that almost half of all internet traffic in 2023 was automated, with bad bots responsible for 32% of total traffic (Imperva, 2024). On platforms like X, automated accounts amplify low-credibility content (Shao et al., 2018), distort political discourse during elections (Bessi and Ferrara, 2016; Howard, Woolley and Calo, 2018), and increase exposure to negative material (Stella, Ferrara and De Domenico, 2018).

As an active social media user I struggle daily with knowing what to believe online. Conversations that look organic can turn out to be manufactured, trending topics can be inflated by coordinated automated accounts, and political opinions in my feed are often shaped by content whose origin I cannot verify (Ferrara et al., 2016; Cresci, 2020). Existing bot detection tools either need academic API access, run as standalone websites the user has to visit manually, or sit behind paywalls. There is no accessible free real-time tool that works inside the browser where the user already is. This project addresses that gap.

### Project Aim
To design, train, and deploy a machine learning system that detects bot accounts on X using only the profile metadata visible to a browser extension, and to deliver it as a working real-time tool that an ordinary user can install and use.

### Project Objectives
1. Acquire and preprocess a public labelled dataset of Twitter accounts.
2. Engineer behavioural features that capture how bots differ from real users.
3. Train and compare three classifier families (RNN, CNN, gradient-boosted trees).
4. Address class imbalance through weighted loss and tune the decision threshold per model.
5. Select the best model by F1 on a held-out test set and serialise it for deployment.
6. Serve the model via a FastAPI backend with a documented prediction endpoint.
7. Build a Chrome extension that scrapes Twitter/X profiles and displays bot scores in real time.
8. Document the system end to end with reproducible code, an analysis notebook, and a presentation.
9. Replace the popup's hand-coded signal rules with real per-prediction feature contributions from the deployed model.
10. Extend the extension with multi-context thread analysis and unsupervised coordinated inauthentic behaviour detection.
11. Validate the deployed model with a custom organisation benchmark and an external architecture-transfer experiment on a published dataset.

### Executive Summary
The XGBoost model trained on 37 engineered numeric features achieved the best performance on a held out test set of 5,616 users: F1 of 0.8076, ROC-AUC of 0.9329, and accuracy of 0.8725. Two deep learning models (BiGRU-LSTM and CNN-BiLSTM) trained on the same data with a gated fusion architecture combining text and numeric branches landed 1.6 and 5.7 F1 points behind XGBoost respectively. The four most discriminative features by gain importance were `verified` (0.20), `followers_count` (0.11), `is_established_account` (0.10), and `log_followers_count` (0.07), with engagement-based ratios such as `favourites_to_statuses_ratio` ranking just below at around 0.03. Log transforms of heavily skewed counts consistently outperformed their raw counterparts on the long-tailed inputs. Profile completeness signals and the four account-type binary flags also contributed despite being simple flags. The result that gradient-boosted trees match deep learning on this kind of tabular data is consistent with the published literature (Shwartz-Ziv and Armon, 2022; Grinsztajn, Oyallon and Varoquaux, 2022). Class-weighted loss combined with per-model threshold tuning added a measurable lift over default settings. The deployed model is around 2.6 MB on disk, trains in under one second, and serves predictions in a few milliseconds from a lightweight FastAPI backend.

Alongside the model metrics, the project covers four product extensions. Per-prediction explainability replaces the popup's hand-coded signal rules with the actual per-feature XGBoost contributions, surfacing the model's own reasoning. A Sentence-BERT bio embedding fusion experiment was tested as a Phase 2 negative result and the deployed model stays at the 37-feature XGBoost baseline. A custom benchmark of 50 well-known verified organisations tests the news-organisation override directly: the model classifies all 50 as HUMAN when given complete metadata, and the override fires meaningfully on only 2 of 50. External architecture-transfer on MGTAB (Liu et al., 2023) reaches F1 = 0.8364, in line with MGTAB's published Random Forest baseline. The Chrome extension was extended with multi-context thread analysis on tweet detail pages, scoring reply authors via a `/predict_batch` endpoint and surfacing aggregate statistics plus coordinated inauthentic behaviour clusters in a sticky panel.

### Audience
This project is for everyday social media users who want a quick way to verify suspicious accounts while browsing, for researchers, journalists, and fact-checkers who need a fast screening tool, and for data science students who want a complete reference example of an end to end machine learning pipeline from data acquisition through to browser deployment.

### Data Sources
The project uses one public labelled dataset of X accounts.

`twitter_human_bots.csv` (airt-ml, 2023). 37,438 X accounts with profile metadata (followers, friends, statuses, favourites, account age, verified status, default profile image flag, profile description, screen name, location) and a binary `account_type` label of human or bot. Hosted on Hugging Face under CC BY-SA 3.0. Downloaded automatically on first run via the `datasets` library.

### Methodology
1. Acquire dataset from Hugging Face and cache to disk.
2. Clean and standardise raw fields, fill missing values, cast types.
3. Engineer 37 numeric features grouped into 8 categories.
4. Random train, validation, test split (70/15/15) with a fixed random seed.
5. Normalise numeric features with a standard scaler fitted on the training split only.
6. Build a 50,000 word vocabulary from the training corpus for the GloVe-based text models.
7. Train BiGRU-LSTM, CNN-BiLSTM, and XGBoost using class-weighted loss to handle the 2:1 imbalance.
8. Tune the decision threshold per model on the validation set.
9. Evaluate on the held out test set using F1, accuracy, precision, recall, ROC-AUC, and PR-AUC.
10. Select the best model by F1 and save it for deployment.
11. Wire the model into a FastAPI backend with a `/predict` endpoint.
12. Build the Chrome extension that scrapes X profiles and calls the backend.
13. Replace the popup's rule-based signals with per-prediction XGBoost feature contributions for real model-grounded explanations.
14. Test a Sentence-BERT bio embedding fusion experiment, retain the 37-feature baseline as the deployed model.
15. Add a `/predict_batch` endpoint and a second content script for tweet detail pages with reply scoring and CIB clustering.
16. Curate a 50-organisation benchmark and validate the news-organisation override at the data level.
17. Train the same XGBoost architecture on MGTAB's pre-computed features as an external architecture-transfer experiment.

### File Directory
```
twitter-bot-detector/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/twitter_human_bots.csv
│   └── processed/processed.pt
│
├── models/
│   └── checkpoints/
│       ├── best_model_info.json
│       ├── bigru_lstm_best.pt
│       ├── cnn_bilstm_best.pt
│       ├── xgboost_best.json
│       └── vocab.json
│
├── src/
│   ├── config.py        paths, hyperparameters, device
│   ├── data.py          download, feature engineering, dataset, vocabulary
│   ├── models.py        BiGRU-LSTM and CNN-BiLSTM with gated fusion
│   ├── train.py         train, evaluate, threshold tune, save best
│   ├── embed_bios.py    pre-compute Sentence-BERT bio embeddings
│   ├── orgs_eval.py     50-organisation custom benchmark
│   └── mgtab_eval.py    MGTAB architecture-transfer evaluation
│
├── api/
│   └── app.py           FastAPI backend with /predict, /predict_batch, /health
│
├── extension/
│   ├── manifest.json
│   ├── background/service-worker.js
│   ├── content/
│   │   ├── content.js   single-profile scraping (profile pages)
│   │   ├── content.css
│   │   ├── thread.js    multi-context reply scoring + CIB clustering (tweet pages)
│   │   └── thread.css
│   ├── popup/
│   └── icons/
│
└── notebooks/
    └── analysis.ipynb   EDA, feature analysis, model comparison, demo
```

### Data Dictionary

#### Raw Fields from `twitter_human_bots.csv`
| Column Name             | Data Type | Description |
|-------------------------|-----------|-------------|
| id                      | int64     | Unique Twitter user ID |
| screen_name             | object    | Twitter handle |
| description             | object    | Profile bio text |
| location                | object    | Self-reported location string |
| created_at              | datetime  | Account creation timestamp |
| followers_count         | int64     | Number of accounts following the user |
| friends_count           | int64     | Number of accounts the user follows |
| statuses_count          | int64     | Total tweets posted |
| favourites_count        | int64     | Total tweets the user has liked |
| verified                | bool      | Whether the account is verified |
| default_profile         | bool      | Whether the user has the default profile theme |
| default_profile_image   | bool      | Whether the user has the default profile image |
| account_age_days        | int64     | Days since account creation |
| average_tweets_per_day  | float64   | Mean tweets per day across the account lifetime |
| account_type            | object    | Target label: `human` or `bot` |

#### Engineered Features (37 total)
After preprocessing and feature engineering, the model receives 37 numeric features grouped into 8 categories. All engineered features are normalised using a z-score scaler fitted on training data only to avoid data leakage.

| Category | Features | Count |
|----------|----------|-------|
| Raw counts | followers_count, friends_count, statuses_count, favourites_count, account_age_days, average_tweets_per_day | 6 |
| Log transforms | log versions of skewed counts (handles 0 to 100M+ range) | 6 |
| Behavioural ratios | followers_to_friends, favourites_to_statuses, statuses_to_followers, friends_to_followers | 4 |
| Profile completeness | verified, default_profile, default_profile_image, has_description, has_location, profile_completeness, description_length, screen_name_length | 8 |
| Screen name patterns | screen_name_digits, screen_name_digit_ratio, screen_name_has_underscore | 3 |
| Activity anomalies | tweets_per_follower, tweets_per_day_per_follower | 2 |
| Bio content signals | bio_url_count, bio_hashtag_count, bio_mention_count, bio_word_count | 4 |
| Account-type signals | bio_has_news_keywords, bio_has_org_keywords, bio_likely_organisation, is_established_account | 4 |

The full ordered list of features is defined in `src/config.py` as `numeric_features`.

### Model Architectures
Three classifiers were trained and compared.

BiGRU-LSTM with gated fusion. A bidirectional GRU followed by an LSTM operating on word embeddings of the user's bio. The text branch output is projected to a fused dimension and combined with a deep `NumericNet` branch over the 37 numeric features through a learned sigmoid gate. The gate decides per-sample whether to lean on the text or the numeric features.

CNN-BiLSTM with gated fusion. A multi-kernel 1D CNN (kernel sizes 3, 4, 5) extracts local n-gram features from the bio embeddings, followed by max-over-time pooling and a bidirectional LSTM. The same `NumericNet` branch and gated fusion are used.

XGBoost. A gradient-boosted decision tree ensemble trained directly on the 37 engineered numeric features. Class imbalance is handled via `scale_pos_weight = 2.03`.

All three use class-weighted loss (or `scale_pos_weight` for XGBoost), early stopping on validation F1, and a per-model decision threshold tuned on the validation set.

### Results

| Model | F1 | Accuracy | Precision | Recall | ROC-AUC | PR-AUC | Threshold | Train Time |
|-------|------|------|------|------|------|------|------|------|
| XGBoost | 0.8076 | 0.8725 | 0.8336 | 0.7832 | 0.9329 | 0.9031 | 0.580 | 1 s |
| BiGRU-LSTM | 0.7914 | 0.8549 | 0.7777 | 0.8056 | 0.9220 | 0.8872 | 0.520 | 829 s |
| CNN-BiLSTM | 0.7503 | 0.8292 | 0.7497 | 0.7509 | 0.9001 | 0.8486 | 0.570 | 49 s |

XGBoost reaches the highest F1 (0.8076) and accuracy (0.8725) on the test set, with BiGRU-LSTM 1.6 points lower on F1 and CNN-BiLSTM 5.7 points lower. Training time differs by three orders of magnitude: 1 second for XGBoost vs 829 seconds for BiGRU-LSTM on the same hardware. ROC-AUC sits between 0.9001 and 0.9329 across all three, so the underlying ranking quality is similar. A Random Forest baseline trained on the same 37 features puts log-transformed counts (`log_followers_count`, `log_favourites_count`, `log_friends_count`) at the top of the importance ranking, with raw counts close behind and engagement ratios such as `favourites_to_statuses_ratio` in the middle of the table at around 0.07. XGBoost itself weights `verified` highest at 0.20 followed by raw `followers_count` at 0.11 and the engineered `is_established_account` flag at 0.10.

Full charts, ROC curves, confusion matrices, and interpretations of each figure are found in `notebooks/analysis.ipynb`.

### Real Model Explainability
The original popup `signals` were a hand-coded `if`/`elif` rule list disconnected from the model. The new implementation calls XGBoost's `pred_contribs` on every prediction and returns the actual per-feature contribution to the log-odds output. Every feature also carries a plain-English description (held in the `feature_descriptions` dict in `api/app.py`) so a non-technical user reads "likes given per tweet posted" instead of `favourites_to_statuses_ratio`.

Three surfaces show the same explanation:

1. **Inline panel injected directly onto profile pages.** When you open any X profile, the content script (`extension/content/content.js` `injectRichPanel()`) embeds a dark card below the user header with the score, the HUMAN/UNCERTAIN/BOT badge, a horizontal score legend bar showing where the score sits on the spectrum, the news-organisation override banner if applicable, and two columns side by side showing the top four contributions toward bot and toward human. Each contribution is a horizontal bar with the plain-English description and the contribution magnitude. This is the primary surface and requires no clicks.
2. **Popup view via the toolbar icon.** The popup mirrors the inline panel's layout in a more compact form factor with a custom dark scrollbar. Because Chrome popups have a hard size cap of around 800x600 pixels, the popup also includes an "open full view in new tab" button at the top that caches the latest analysis in `chrome.storage.local` and opens the same view in a new tab where the viewport is unlimited (`body.fullview` CSS class triggers the larger layout).
3. **Hover cards on reply dots in tweet threads.** Each scored reply gets a coloured dot next to its username; hovering reveals a 260-pixel card with the same score, badge, score legend, and top three contributions per side in plain English.

For Kim Kardashian, the strongest pull toward HUMAN is `has the verified blue checkmark` at -0.77 followed by `tweets per day relative to followers` at -0.70 and `total followers` at -0.57. For Al Jazeera English, the model's strongest signals point toward bot (`tweets posted per day on average` at +1.43, `total tweets posted` at +0.56), and the news-organisation override caps the probability at `threshold - 0.15` (around 0.43) to keep the final label HUMAN. Implementation is in `compute_contributions()` and `format_feature_value()` in `api/app.py`, the inline panel rendering in `extension/content/content.js`, the popup rendering in `extension/popup/popup.js`, and the thread hover cards in `extension/content/thread.js`.

### Multi-Context Thread Analysis
A second content script (`extension/content/thread.js`) loads only on tweet detail pages (`https://x.com/*/status/*`). It uses a `MutationObserver` plus a 30-attempt polling fallback to scrape reply authors as the user scrolls, batches them into a `/predict_batch` API call (up to 50 profiles per call), injects coloured dots inline next to each username, and renders a sticky panel pinned in the bottom-right corner of the viewport. The panel shows live status text such as "scanning... 56 cells (cellInnerDiv)", transitions to "N scored" once results return, and includes per-class counts of HUMAN / UNCERTAIN / BOT plus suspicious cluster banners. The panel is positioned with `z-index: 2147483647` so it cannot be hidden behind X's right sidebar.

Hovering any reply dot reveals a rich 260-pixel hover card with the same content as the popup and the inline panel: score, badge, score legend bar, and the top three contributions per side in plain English with horizontal bars. This means every reply in a thread has the same explainability as a single-profile lookup.

The popup also has a thread mode. When you click the toolbar icon while on a tweet detail page, the popup queries `thread.js` via a `GET_THREAD_STATS` message and shows the same totals, a stacked horizontal bar of human/uncertain/bot percentages, and any detected clusters. If `thread.js` is loaded but no replies have been scored yet (e.g. you opened the popup before scrolling), the popup shows a clear "loaded but no replies scored yet" message; if `thread.js` did not load at all (e.g. the extension was not reloaded after an update), it shows explicit reload instructions.

Twitter is a single-page app, so navigating from a thread page to a profile page does not reload the document. `thread.js` listens for URL changes via the same `MutationObserver` and runs a `cleanupPanel()` step that removes the sticky panel and resets all state when the new URL is no longer a thread page, plus a `resetState()` step that clears the scored map and re-runs scanning when the new URL is a different thread page. This avoids stale state leaking between pages.

The CIB detector uses greedy single-linkage agglomerative clustering on per-reply features (handle length, digit ratio, ends-in-digits flag, and Jaccard overlap on handle token sets). Clusters with at least 3 members are reported, and a cluster is flagged as suspicious if at least two of these hold: all members end in 3+ digits, all members are scored bot or uncertain, all members have digit ratio above 0.2. The CIB detector does not learn coordination from labelled data; it is heuristic clustering on visible features intended to surface candidates an analyst should investigate.

The reply DOM on Twitter/X exposes only username, display name, and verified status. Per-reply scoring is therefore approximate, since the model expects all 37 engineered features and the missing fields fall back to defaults. The aggregate summary panel and the CIB cluster banners use only the visible features and are the more reliable outputs of the multi-context view. Improving DOM scraping (e.g. recovering follower count and account age from hover cards) is the largest single follow-up for the deployed product.

### Custom Benchmark on 50 Verified Organisations
A hand-curated benchmark of 50 well-known verified organisations across news (15), tech (10), retail and brands (10), sports (8), and NGOs and government (7) tests how the model handles legitimate organisation accounts. The model classifies all 50 as HUMAN on its own when given complete profile metadata. No organisation scores above 50/100 without the override. The news-organisation override fires meaningfully on only 2 of 50 accounts (NFL and F1), nudging their scores from 47 and 48 to 43.

This shows the deployed model handles organisation accounts correctly when given full metadata and the override is a defensive safety net rather than a primary mechanism. The live extension misclassifications observed earlier on Al Jazeera (84/100) and CNN (92/100) were caused by incomplete metadata in the DOM scraping path of `content.js`, not by the model's behaviour on full input. Implementation in `src/orgs_eval.py`, results in `models/checkpoints/orgs_eval_results.json`.

### External Validation on MGTAB
MGTAB (Liu et al., 2023, arXiv:2301.01123) is a published bot detection dataset with 10,199 expert-annotated users and a multi-relational graph. The dataset ships only as preprocessed PyTorch tensors with 788 pre-extracted features per user, not raw profile fields, so the deployed model cannot run on it directly. We trained the same XGBoost architecture on MGTAB's pre-computed features as an architecture-transfer experiment.

| Model | F1 on MGTAB |
|-------|-------------|
| Random Forest (Liu et al., 2023) | ~0.84 |
| GCN (Liu et al., 2023) | ~0.86 |
| RGT graph model (Liu et al., 2023) | ~0.89 |
| Our XGBoost (this work) | 0.8364 |

Our XGBoost reaches F1 = 0.8364, sitting alongside MGTAB's published Random Forest baseline (~0.84). The graph-based models reach a few points higher because MGTAB's contribution is its multi-relational graph, which feature-only models cannot use. This confirms that the XGBoost architecture transfers across datasets when given comparable features. Implementation in `src/mgtab_eval.py`, results in `models/checkpoints/mgtab_eval_results.json`.

### Conclusion
Industry-level bot detection systems used by companies like X Corp. and Meta normally use graph neural networks operating on the full social graph. Those approaches require infrastructure access that those companies have but external tools do not. This project operates within the constraints of a browser extension: only what a user sees on a profile page is available to the model. Within that constraint, gradient-boosted trees on engineered tabular features are competitive with deep learning, consistent with the published literature.

The trained model deploys end to end. When a user opens any profile, the content script scrapes the visible profile metadata, sends it to a FastAPI backend, runs the XGBoost model with a tuned threshold, and returns a bot probability score plus the model's actual per-feature contributions in plain English. The contributions render in three places without any extra interaction: an inline rich panel injected directly into the profile page header, the toolbar popup (with a "open full view in new tab" escape hatch for Chrome's popup size cap), and rich hover cards on every reply dot in tweet threads. No academic API access is needed and no manual data collection is required by the user.

### Recommendations
1. Use the extension as a quick screen, not a final verdict. A score above 80 is a strong signal but is still inconclusive. Manual inspection should still happen for high stakes decisions.
2. Prefer tree-based models for tabular bot features. Deep learning is worth the cost only when you have rich text, image, or graph data. This is consistent with Shwartz-Ziv and Armon (2022) and Grinsztajn, Oyallon and Varoquaux (2022).
3. Engineer behavioural ratios, not just raw counts. The biggest single accuracy gains came from feature engineering, not from architectural changes.
4. Tune decision thresholds per model. Default thresholds of 0.5 leave performance on the table when classes are imbalanced.
5. Plan around the inference-time constraint. A browser extension can only see what is rendered on the profile page. Any future improvement that depends on the social graph or full tweet history will require backend infrastructure that the extension cannot provide on its own.

### Areas for Further Research
1. Tweet content features. The current dataset only has profile bios, not tweet histories. Training on a dataset with real tweet text would let the text models contribute more meaningfully.
2. Graph neural networks. On TwiBot-22 (Feng et al., 2022) and MGTAB (Liu et al., 2023), graph neural networks sit a few F1 points above feature-only models. Replicating these methods would require follower graph data that is not available to a browser extension at inference time.
3. Cross-platform generalisation. Testing whether the same engineered features and architecture transfer to Instagram, Threads, Reddit, or YouTube would strengthen the case for the chosen feature set.
4. Modern LLM-driven bots. Newer bots powered by large language models can produce convincingly human bios and posts. Evaluating the model against these adversarial examples would be a useful robustness test.
5. Temporal behavioural signals. Even without graph access, the extension could in principle scrape relative timestamps from visible tweets and compute spacing patterns. Adding this would give the model a temporal signal it currently lacks.

### Limitations and Improvements

#### Limitations
The dataset and the extension both rely on profile-level features. Full tweet content, follow graph, and posting timestamps are not available. The model was trained on one source, and bot definitions vary across datasets. Bot tactics evolve faster than the dataset, so labels may not match the most modern bot classes. The extension can only see what the user sees on the profile page. The chosen decision threshold trades off precision against recall, so different use cases may want different thresholds.

The training set predates ChatGPT-style accounts and the model was not validated against modern LLM-driven bots. This is a real and unsolved gap in the public bot detection literature, not a project failure. The Sentence-BERT bio embedding fusion experiment in Phase 2 produced a small regression (F1 -0.0025), confirming that on this dataset with these engineered features, modern transformer text encoders do not improve gradient-boosted trees. The most impactful follow-up for the deployed product is improving `content.js` DOM scraping to recover more profile fields (followers, age, statuses count) from the visible profile or hover card. The 50-organisation benchmark in this README shows the model itself classifies all 50 correctly when given complete metadata, so the live failures observed on Al Jazeera and CNN trace back to the scraper, not the classifier.

#### Improvements for Future Work
Add tweet text scraping in the extension by updating the content script to extract visible tweets and send them with the prediction request. Ensemble XGBoost with the neural models for a small but consistent F1 lift. Add a model staleness check that periodically retrains on newer labelled data. Add user feedback in the extension so users can mark predictions as correct or incorrect, which builds a labelled feedback set for retraining. Build a small test set of modern LLM-driven bots and measure the model's robustness against them.

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

Reimers, N. and Gurevych, I. (2019). 'Sentence-BERT: Sentence embeddings using Siamese BERT-networks'. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 3982 to 3992.

Pennington, J., Socher, R. and Manning, C. (2014). 'GloVe: Global vectors for word representation'. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 1532 to 1543.

Shao, C., Ciampaglia, G.L., Varol, O., Yang, K.C., Flammini, A. and Menczer, F. (2018). 'The spread of low-credibility content by social bots'. *Nature Communications*, 9(1), p. 4787.

Shwartz-Ziv, R. and Armon, A. (2022). 'Tabular data: Deep learning is not all you need'. *Information Fusion*, 81, pp. 84 to 90.

Stella, M., Ferrara, E. and De Domenico, M. (2018). 'Bots increase exposure to negative and inflammatory content in online social systems'. *Proceedings of the National Academy of Sciences*, 115(49), pp. 12435 to 12440.