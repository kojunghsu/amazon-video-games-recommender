# Amazon Video Games Recommendation System

A personalized recommendation system built on the [Amazon Reviews 2023 — Video Games](https://amazon-reviews-2023.github.io/data_processing/5core.html) dataset. The project implements and compares **Matrix Factorization (MF)** and **Neural Collaborative Filtering (NCF)** in PyTorch, using a time-aware evaluation setup and user-segment performance analysis.

---

## Business Problem

Video game catalogs on platforms like Amazon contain tens of thousands of titles. Without personalization, users face discovery friction, and the platform risks lower engagement and conversion. A collaborative filtering system can learn from historical ratings and surface games a user is likely to enjoy, even when product metadata is unavailable.

---

## Dataset

**Source:** [Amazon Reviews 2023 — Video Games (5-core)](https://amazon-reviews-2023.github.io/data_processing/5core.html)  
**Citation:** Hou et al. (2024), McAuley Lab, UC San Diego

The 5-core variant retains only users and items with at least five interactions, making it a clean baseline for collaborative filtering research.

| Property | Value |
|---|---:|
| Full dataset interactions | 814,586 |
| Full dataset unique users | 94,762 |
| Full dataset unique items | 25,612 |
| Rating scale | 1.0–5.0 explicit ratings |
| Full dataset date range | Oct 1999–Sep 2023 |
| Modeling subset | 200,000 interactions, reproducible random sample with `SEED=42` |
| Modeling subset unique users | 80,521 |
| Modeling subset unique items | 23,588 |

Each record contains four core fields: `user_id`, `parent_asin` (renamed to `item_id`), `rating`, and `timestamp` in milliseconds since epoch.

> The raw data file is not included in this repository. See [Setup & Usage](#setup--usage) for download instructions. The data is made available for academic and non-commercial research use by the original authors.

---

## Repository Structure

```text
amazon-video-games-recommender/
├── amazon_video_games_recommender.ipynb   # Main notebook, run top to bottom
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Excludes local data, cache, and notebook artifacts
├── LICENSE                                # MIT License
└── README.md
```

---

## Methodology

### 1. Sampling and Preprocessing

The full 814,586-record dataset is reduced to a reproducible **200,000-interaction random sample** using `SEED=42`. This keeps training manageable while preserving enough scale for collaborative filtering.

After sampling, the workflow:

- Keeps `user_id`, `item_id`, `rating`, `timestamp`, and `timestamp_dt`
- Drops rows with missing user, item, rating, or timestamp values
- Casts `rating` to float
- Sorts by timestamp
- Resolves duplicate `(user_id, item_id)` pairs by keeping the most recent interaction

In the current run, no duplicate user-item pairs were removed.

### 2. Temporal Train/Test Split

The split is time-aware rather than random:

1. Draw a reproducible 200,000-interaction random sample from the full dataset
2. Sort the sampled interactions by `timestamp`
3. Use the earliest 80% as training data
4. Use the latest 20% as test data

| Split | Interactions | Date range |
|---|---:|---|
| Training | 160,000 | Oct 1999–Dec 9, 2019 |
| Test | 40,000 | Dec 9, 2019–Aug 29, 2023 |

This mirrors deployment more closely than a random split: the model learns from historical behavior and is evaluated on future interactions.

### 3. Cold-Start Handling

MF and NCF both rely on learned user and item ID embeddings. If a user or item appears for the first time in the test period, the model has no learned embedding for that ID.

The notebook therefore identifies cold-start cases and evaluates model accuracy only on the **warm-start test subset**, where both the user and item appeared during training.

| Metric | Value |
|---|---:|
| Test interactions | 40,000 |
| Test users unseen during training | 12,707 |
| Test items unseen during training | 5,418 |
| Cold-start test interactions | 32,160 |
| Cold-start percentage | 80.40% |
| Warm-start test interactions used for RMSE/MAE | 7,840 |
| Warm-start percentage | 19.60% |

In production, cold-start cases would require fallback strategies such as popularity-based recommendations, genre/metadata-based recommendations, or onboarding preference collection.

### 4. Models

#### Matrix Factorization (MF)

Each user and item is represented by a 32-dimensional embedding. The predicted rating is:

```text
ŷ(u, i) = μ + b_u + b_i + p_u · q_i
```

where `μ` is the global mean rating, `b_u` and `b_i` are user/item bias terms, and `p_u · q_i` is the dot product of the user and item embeddings.

Configuration:

- Embedding dimension: 32
- Optimizer: Adam
- Learning rate: 0.005
- Weight decay: 1e-5
- Loss: MSE
- Max epochs: 20
- Early stopping patience: 3

#### Neural Collaborative Filtering (NCF)

NCF concatenates user and item embeddings and passes them through a three-layer MLP:

```text
[user_embedding || item_embedding]
→ Linear(64 → 128) → ReLU → Dropout(0.2)
→ Linear(128 → 64) → ReLU → Dropout(0.1)
→ Linear(64 → 1)
```

Configuration:

- Embedding dimension: 32
- Optimizer: Adam
- Learning rate: 0.001
- Weight decay: 1e-5
- Loss: MSE
- Max epochs: 20
- Early stopping patience: 3

Both models use a random 90/10 split of the training data for model training and validation. This validation split is used only for early stopping and checkpoint selection. Final model performance is reported on the time-based warm-start test set. Predictions are clipped to the valid 1.0–5.0 rating range before evaluation.

---

## Results

Results below come from the notebook output with `SEED=42` on CPU. Exact values may vary slightly by PyTorch or hardware version.

### Overall Warm-Start Test Performance

| Model | RMSE | MAE | Early stopping |
|---|---:|---:|---|
| **Matrix Factorization** | **1.3088** | **1.0188** | Stopped at epoch 7; best validation epoch 4 |
| Neural Collaborative Filtering | 1.3797 | 1.0584 | Stopped at epoch 6; best validation epoch 3 |

MF outperforms NCF on both RMSE and MAE. A likely explanation is sparsity: many users have very few ratings, making it difficult for the neural model to learn stable non-linear preference patterns. The simpler latent-factor structure of MF generalizes more consistently in this setting.

### User-Segment Performance

Warm-start test interactions are segmented into tertiles based on each user's number of interactions in the **full training set**. The tertile thresholds are computed on the distribution of training-interaction counts within the warm-start test population.

Segmentation criteria from the notebook:

- Light users: training interactions <= 1
- Average users: training interactions > 1 and <= 2
- Power users: training interactions > 2

| Segment | Test interactions | Avg train interactions per user | MF RMSE | MF MAE | NCF RMSE | NCF MAE |
|---|---:|---:|---:|---:|---:|---:|
| Light users | 3,740 | 1.0000 | 1.3206 | 1.0350 | 1.4065 | 1.1281 |
| Average users | 1,869 | 2.0000 | 1.3091 | 1.0165 | 1.3653 | 1.0297 |
| Power users | 2,231 | 5.6132 | 1.2884 | 0.9937 | 1.3459 | 0.9657 |

The segment analysis shows that prediction quality improves as users have more historical interactions. MF has lower RMSE in all three segments and lower MAE for light and average users. NCF achieves lower MAE for power users, suggesting that a more flexible model may become useful when richer user histories are available, although MF remains the more stable model overall in this experiment.

---

## Limitations

- **Cold-start coverage:** 80.40% of test interactions involve unseen users or items and cannot be directly evaluated by ID-embedding models.
- **Data sparsity:** many users have limited rating histories, which constrains personalization quality.
- **Explicit feedback only:** ratings reflect post-purchase opinions and do not include browsing, search, cart, or wishlist behavior.
- **No item metadata:** the models do not use genre, platform, developer, price, or text features, which limits cold-start handling and explainability.

---

## Future Improvements

- Add item metadata such as genre, platform, developer, and product text for hybrid recommendations.
- Build fallback logic for cold-start users and items using popularity, metadata, or onboarding questions.
- Explore sequence-aware models such as SASRec to capture evolving user preferences.
- Incorporate implicit feedback signals such as views, clicks, wishlists, or purchases.
- Retrain on a rolling time window to keep embeddings current as new users and games appear.

---

## Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/amazon-video-games-recommender.git
cd amazon-video-games-recommender
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU-accelerated training, follow the official PyTorch installation instructions for your CUDA version. The notebook automatically falls back to CPU if GPU is unavailable.

### 3. Download the dataset

1. Visit the Amazon Reviews 2023 5-core dataset page.
2. Download the **Video Games** 5-core CSV file.
3. Place it in the repository root with this exact filename:

```text
Video_Games.csv.gz
```

The `.gitignore` excludes this raw data file so it is not accidentally committed.

### 4. Run the notebook

```bash
jupyter notebook amazon_video_games_recommender.ipynb
```

Then run all cells from top to bottom. On CPU, training time depends on hardware, but the 200,000-interaction sample is designed to keep the workflow manageable.

---

## Tech Stack

| Library | Role |
|---|---|
| `pandas` | Data loading, cleaning, timestamp handling, and analysis |
| `numpy` | Numerical operations and reproducibility |
| `scikit-learn` | RMSE/MAE calculation |
| `torch` | Model definition, training loop, embeddings, and early stopping |
| `torch.utils.data` | Dataset and DataLoader for batched training/evaluation |
| `notebook` | Interactive execution environment |

---

## License

This project is released under the MIT License. See `LICENSE` for details.

---

## Acknowledgements

Dataset: Hou et al. (2024), *Bridging Language and Items for Retrieval and Recommendation*, McAuley Lab, UC San Diego.
