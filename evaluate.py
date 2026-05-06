# === FILE: evaluate.py ===
# === PART OF: Movie Recommender System ===
# === IMPORTS FROM: cf_model (svd_predictions, knn_predictions) ===

from collections import defaultdict

from surprise import accuracy
from cf_model import svd_predictions, knn_predictions


# ============================================================
# Compute RMSE and MAE for both SVD and KNN models
# ============================================================

svd_rmse = accuracy.rmse(svd_predictions, verbose=False)
svd_mae  = accuracy.mae(svd_predictions, verbose=False)
knn_rmse = accuracy.rmse(knn_predictions, verbose=False)
knn_mae  = accuracy.mae(knn_predictions, verbose=False)


# ============================================================
# Precision@K and Recall@K helpers
# ============================================================

# Group predictions by user and return top N per user
def get_top_n(predictions, n=5):
    """
    From a list of Surprise predictions, group by user and
    return the top N highest-predicted items for each user.
    """
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est, true_r))

    for uid in top_n:
        top_n[uid].sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = top_n[uid][:n]

    return top_n


# Compute average Precision@K and Recall@K across all users
def precision_recall_at_k(predictions, k=5, threshold=3.5):
    """
    Compute average Precision@K and Recall@K across all users.

    - A 'relevant' item is one where the actual rating >= threshold.
    - Precision@K = (relevant items in top-K recommendations) / K
    - Recall@K    = (relevant items in top-K) / (total relevant items for user)
    """
    user_est = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est[uid].append((iid, est, true_r))

    precisions = []
    recalls    = []

    for uid, preds in user_est.items():
        preds.sort(key=lambda x: x[1], reverse=True)
        top_k = preds[:k]

        n_relevant_in_k = sum(1 for (_, _, true_r) in top_k if true_r >= threshold)
        n_relevant_total = sum(1 for (_, _, true_r) in preds if true_r >= threshold)

        precisions.append(n_relevant_in_k / k)
        recalls.append(
            n_relevant_in_k / n_relevant_total if n_relevant_total > 0 else 0
        )

    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall    = sum(recalls) / len(recalls) if recalls else 0

    return round(avg_precision, 4), round(avg_recall, 4)


# ============================================================
# Pre-compute Precision@5 and Recall@5 for both models
# ============================================================

svd_prec, svd_rec = precision_recall_at_k(svd_predictions, k=5, threshold=3.5)
knn_prec, knn_rec = precision_recall_at_k(knn_predictions, k=5, threshold=3.5)


# ============================================================
# Build and return the full evaluation report as a dict
# ============================================================

# Return all evaluation metrics in a single dictionary
def get_evaluation_report():
    """
    Return a dict with all evaluation metrics for SVD and KNN.
    Keys: svd_rmse, svd_mae, knn_rmse, knn_mae, precision_at_5, recall_at_5
    """
    return {
        "svd_rmse":      round(svd_rmse, 4),
        "svd_mae":       round(svd_mae, 4),
        "knn_rmse":      round(knn_rmse, 4),
        "knn_mae":       round(knn_mae, 4),
        "precision_at_5": svd_prec,
        "recall_at_5":    svd_rec,
        # Also include per-model precision/recall for completeness
        "SVD":  {"RMSE": round(svd_rmse, 4), "MAE": round(svd_mae, 4),
                 "Precision@5": svd_prec, "Recall@5": svd_rec},
        "KNN":  {"RMSE": round(knn_rmse, 4), "MAE": round(knn_mae, 4),
                 "Precision@5": knn_prec, "Recall@5": knn_rec},
    }
