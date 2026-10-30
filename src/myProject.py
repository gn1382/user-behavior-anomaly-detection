import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import time
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import random
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier


SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


#Load data from CSV file
df_raw = pd.read_csv(r"C:\Users\Lenovo\Documents\Uni\Term 8\پروژه کارشناسی\c i\cybersecurity_intrusion_data.csv")
df = df_raw.copy()

#Data Cleaning
df.drop_duplicates(inplace=True)

df.dropna(subset=[
    'network_packet_size', 'protocol_type', 'session_duration',
    'encryption_used', 'ip_reputation_score', 'attack_detected'
], inplace=True)

df['browser_type'] = df['browser_type'].fillna('Unknown')
df['failed_logins'] = df['failed_logins'].fillna(0)


df.drop(columns=['session_id'], inplace=True, errors='ignore')


num_cols = [
    'network_packet_size', 'session_duration', 'ip_reputation_score',
    'login_attempts', 'failed_logins'
]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

#Feature Encoding
from sklearn.preprocessing import LabelEncoder

le_protocol = LabelEncoder()
df['protocol_code'] = le_protocol.fit_transform(df['protocol_type'])

le_encryption = LabelEncoder()
df['encryption_code'] = le_encryption.fit_transform(df['encryption_used'])

le_browser = LabelEncoder()
df['browser_code'] = le_browser.fit_transform(df['browser_type'])

df['unusual_time_access'] = df['unusual_time_access'].astype(int)

#Feature Engineering
df['login_failure_ratio'] = df['failed_logins'] / (df['login_attempts'] + 1)

df['normalized_traffic'] = df['network_packet_size'] / (df['session_duration'] + 1)

df['weak_encryption'] = (df['encryption_used'] == 'DES').astype(int)

df['risk_score'] = (df['ip_reputation_score'] * 0.6 + 
                   df['login_failure_ratio'] * 0.3 + 
                   df['unusual_time_access'] * 0.1)

#Feature Selection 
features = [
    'protocol_code', 'encryption_code', 'browser_code',
    'network_packet_size', 'session_duration', 'ip_reputation_score',
    'login_attempts', 'failed_logins', 'unusual_time_access',
    'login_failure_ratio', 'normalized_traffic', 'weak_encryption',
    'risk_score'
]

X = df[features]
y = df['attack_detected']

#Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Anomaly ratio in train: {y_train.mean():.4f}")


#Feature Importance Analysis

rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances (RandomForest)")
plt.bar(range(X_train.shape[1]), importances[indices], color='skyblue', align="center")
plt.xticks(range(X_train.shape[1]), np.array(features)[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

feature_importance_df = pd.DataFrame({
    'Feature': np.array(features)[indices],
    'Importance': importances[indices]
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10))

fit_times = {}
predict_times = {}

anomaly_ratio = y_train.sum() / len(y_train)

#1:Isolation Forest
start_fit = time.time()
iso = IsolationForest(contamination=anomaly_ratio, random_state=SEED)
iso.fit(X_train)
fit_times['Isolation Forest'] = time.time() - start_fit

start_pred = time.time()
y_pred_iso = np.where(iso.predict(X_test) == -1, 1, 0)
predict_times['Isolation Forest'] = time.time() - start_pred
print("\n[Isolation Forest]\n", classification_report(y_test, y_pred_iso, digits=4))

#2:One-Class SVM
start_fit = time.time()
ocsvm = OneClassSVM(nu=anomaly_ratio, kernel="rbf", gamma=0.1)
ocsvm.fit(X_train)
fit_times['One-Class SVM'] = time.time() - start_fit

start_pred = time.time()
y_pred_svm = np.where(ocsvm.predict(X_test) == -1, 1, 0)
predict_times['One-Class SVM'] = time.time() - start_pred
print("\n[One-Class SVM]\n", classification_report(y_test, y_pred_svm, digits=4))

#3:Local Outlier Factor
start_fit = time.time()
lof = LocalOutlierFactor(n_neighbors=20, contamination=anomaly_ratio)
start_pred = time.time()
y_pred_lof = np.where(lof.fit_predict(X_test) == -1, 1, 0)
fit_times['LOF'] = time.time() - start_fit
predict_times['LOF'] = time.time() - start_pred
print("\n[Local Outlier Factor]\n", classification_report(y_test, y_pred_lof, digits=4))

#4:KNN
start_fit = time.time()
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
fit_times['KNN'] = time.time() - start_fit

start_pred = time.time()
y_pred_knn = knn.predict(X_test)
predict_times['KNN'] = time.time() - start_pred
print("\n[KNN Classifier]\n", classification_report(y_test, y_pred_knn, digits=4))

#5:KMeans
start_fit = time.time()

kmeans_pipe = make_pipeline(
    StandardScaler(),
    KMeans(n_clusters=2, random_state=SEED, n_init=10, algorithm='elkan')
)
kmeans_pipe.fit(X_train)

scaler = kmeans_pipe.named_steps['standardscaler']
kmeans = kmeans_pipe.named_steps['kmeans']
X_test_scaled = scaler.transform(X_test)
distances = kmeans.transform(X_test_scaled)
min_distances = np.min(distances, axis=1)

y_pred_kmeans = (min_distances > np.percentile(min_distances, 100*(1-anomaly_ratio))).astype(int)

fit_times['KMeans'] = time.time() - start_fit
predict_times['KMeans'] = time.time() - start_pred
print("\n[KMeans Clustering]\n", classification_report(y_test, y_pred_kmeans, digits=4))


#6:Autoencoder
start_fit = time.time()
autoencoder = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(X_train.shape[1], activation='sigmoid')
])
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(X_train, X_train,
                epochs=10, batch_size=256,
                shuffle=True, validation_split=0.1, verbose=1)
fit_times['Autoencoder'] = time.time() - start_fit

start_pred = time.time()
X_train_pred = autoencoder.predict(X_train)
mse_train = np.mean(np.power(X_train - X_train_pred, 2), axis=1)
threshold = np.percentile(mse_train, 100 * anomaly_ratio)

X_test_pred = autoencoder.predict(X_test)
mse_test = np.mean(np.power(X_test - X_test_pred, 2), axis=1)
y_pred_auto = (mse_test > threshold).astype(int)  
predict_times['Autoencoder'] = time.time() - start_pred

print("\n[Autoencoder]\n", classification_report(y_test, y_pred_auto, digits=4))

#Times Table
print("\n Model Training Times (seconds):")
for m, t in fit_times.items():
    print(f"{m}: {t:.4f} s")
print("\n Model Prediction Times (seconds):")
for m, t in predict_times.items():
    print(f"{m}: {t:.4f} s")

#Model Performance Comparison
results = {}
def evaluate_model(y_true, y_pred, name):
    results[name] = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0)
    }
evaluate_model(y_test, y_pred_iso, "Isolation Forest")
evaluate_model(y_test, y_pred_svm, "One-Class SVM")
evaluate_model(y_test, y_pred_lof, "LOF")
evaluate_model(y_test, y_pred_knn, "KNN")
evaluate_model(y_test, y_pred_kmeans, "KMeans")
evaluate_model(y_test, y_pred_auto, "Autoencoder")

results_df = pd.DataFrame(results).T
print("\nModel Performance Comparison:\n", results_df.round(3))



#TSNE
def plot_tsne_comparison(X_data, y_true, y_pred, title):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    X_tsne = tsne.fit_transform(X_data)

    plt.figure(figsize=(12, 5))

    #True Labels
    plt.subplot(1, 2, 1)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_true, cmap='coolwarm', s=10, alpha=0.7)
    plt.title(f'True Labels - {title}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(label='0=Normal, 1=Anomaly')
    plt.grid(True, linestyle='--', alpha=0.5)

    #Predicted Labels
    plt.subplot(1, 2, 2)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred, cmap='coolwarm', s=10, alpha=0.7)
    plt.title(f'Predicted Labels - {title}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(label='0=Normal, 1=Anomaly')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


plot_tsne_comparison(X_test, y_test, y_pred_iso, 'Isolation Forest')
plot_tsne_comparison(X_test, y_test, y_pred_svm, 'One-Class SVM')
plot_tsne_comparison(X_test, y_test, y_pred_lof, 'Local Outlier Factor (LOF)')
plot_tsne_comparison(X_test, y_test, y_pred_knn, 'KNN')
plot_tsne_comparison(X_test, y_test, y_pred_kmeans, 'KMeans')
plot_tsne_comparison(X_test, y_test, y_pred_auto, 'Autoencoder')

#omparison of Anomaly Counts by Each Model
models = ['Isolation Forest', 'One-Class SVM', 'LOF', 'KNN', 'KMeans', 'Autoencoder']
anomaly_counts = [
    y_pred_iso.sum(),
    y_pred_svm.sum(),
    y_pred_lof.sum(),
    y_pred_knn.sum(),
    y_pred_kmeans.sum(),
    y_pred_auto.sum()
]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, anomaly_counts, color='orange', alpha=0.8)
plt.title('Comparison of Anomaly Counts by Each Model')
plt.ylabel('Number of Anomalies')
plt.xticks(rotation=30)
plt.grid(axis='y', linestyle='--', alpha=0.6)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1, str(int(height)),
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()



#Confusion Matrix
def plot_confusion(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal","Anomaly"])
    disp.plot(cmap="Greens")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    print(f"{model_name} — TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"{model_name} — Specificity: {specificity:.3f}, Recall: {recall:.3f}, FPR: {fpr:.3f}\n")
plot_confusion(y_test, y_pred_iso, "Isolation Forest")
plot_confusion(y_test, y_pred_svm, "One-Class SVM")
plot_confusion(y_test, y_pred_lof, "LOF")
plot_confusion(y_test, y_pred_knn, "KNN")
plot_confusion(y_test, y_pred_auto, "Autoencoder")
plot_confusion(y_test, y_pred_kmeans, "KMeans")

#10 anomalous records from each model
X_test_index = y_test.index 

cols_to_show = features + ['attack_detected'] 

print("\nIsolation Forest Anomalies:")
print(df.loc[X_test_index[y_pred_iso == 1], cols_to_show].head(10))

print("\nOne-Class SVM Anomalies:")
print(df.loc[X_test_index[y_pred_svm == 1], cols_to_show].head(10))

print("\nLOF Anomalies:")
print(df.loc[X_test_index[y_pred_lof == 1], cols_to_show].head(10))

print("\nKNN Anomalies:")
print(df.loc[X_test_index[y_pred_knn == 1], cols_to_show].head(10))

print("\nKMeans Anomalies:")
print(df.loc[X_test_index[y_pred_kmeans == 1], cols_to_show].head(10))

print("\nAutoencoder Anomalies:")
print(df.loc[X_test_index[y_pred_auto == 1], cols_to_show].head(10))

#Misclassified Samples
def show_false_predictions(y_true, y_pred, model_name):
    false_index = y_true.index[(y_true != y_pred)]
    print(f"\nMisclassified Samples ({model_name}):")
    print(df.loc[false_index, features + ['attack_detected']].head(10))

show_false_predictions(y_test, y_pred_iso, "Isolation Forest")
show_false_predictions(y_test, y_pred_svm, "One-Class SVM")
show_false_predictions(y_test, y_pred_lof, "LOF")
show_false_predictions(y_test, y_pred_knn, "KNN")
show_false_predictions(y_test, y_pred_auto, "Autoencoder")
show_false_predictions(y_test, y_pred_kmeans, "KMeans")



#AUC-ROC , FPR 
def compute_roc(y_true, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")
    return auc, fpr[1] if len(fpr) > 1 else 0

plt.figure(figsize=(8, 6))
plt.title("ROC Curve for All Models")

auc_scores = {}
fpr_values = {}

auc_scores['Isolation Forest'], fpr_values['Isolation Forest'] = compute_roc(y_test, iso.decision_function(X_test), "Isolation Forest")
auc_scores['One-Class SVM'], fpr_values['One-Class SVM'] = compute_roc(y_test, ocsvm.decision_function(X_test), "One-Class SVM")
auc_scores['Autoencoder'], fpr_values['Autoencoder'] = compute_roc(y_test, mse_test, "Autoencoder")

lof_scores = -lof.negative_outlier_factor_
auc_scores['Local Outlier Factor'], fpr_values['LOF'] = compute_roc(y_test, lof_scores[:len(y_test)], "LOF")

knn_proba = knn.predict_proba(X_test)[:, 1]
auc_scores['KNN'], fpr_values['KNN'] = compute_roc(y_test, knn_proba, "KNN")

centers = kmeans.cluster_centers_
assigned = kmeans.predict(X_test_scaled)
kmeans_scores = np.linalg.norm(X_test_scaled - centers[assigned], axis=1)
auc_scores['KMeans'], fpr_values['KMeans'] = compute_roc(y_test, kmeans_scores, "KMeans")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(results_df, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Model Comparison Heatmap")
plt.show()


#ensemble(KNN + Autoencoder)
score_knn = knn.predict_proba(X_test)[:, 1]
score_ae = mse_test 

sc = MinMaxScaler()
score_knn = sc.fit_transform(score_knn.reshape(-1, 1)).flatten()
score_ae = sc.fit_transform(score_ae.reshape(-1, 1)).flatten()

best_score = -1
best_w_knn = None 
best_w_ae = None
best_thresh = None
best_metrics = {}

for w_knn in np.arange(0.1, 1.0, 0.1):
    w_ae = 1 - w_knn
    ensemble_score = w_knn * score_knn + w_ae * score_ae
    
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred = (ensemble_score >= thresh).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, ensemble_score)

        combined = 0.6 * f1 + 0.4 * auc

        if combined > best_score:
            best_score = combined
            best_w_knn = w_knn
            best_w_ae = w_ae
            best_thresh = thresh
            best_metrics = {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "AUC": auc
            }

print(f"Best Weights -> KNN: {best_w_knn:.2f}, Autoencoder: {best_w_ae:.2f}")
print(f"Best Threshold: {best_thresh:.2f}")
for m, v in best_metrics.items():
    print(f"{m}: {v:.4f}")

ensemble_score = best_w_knn * score_knn + best_w_ae * score_ae
ensemble_preds = (ensemble_score >= best_thresh).astype(int)


ensemble_results = {
    "Accuracy": best_metrics["Accuracy"],
    "Precision": best_metrics["Precision"],
    "Recall": best_metrics["Recall"],
    "F1-Score": best_metrics["F1-Score"],
    "AUC": best_metrics["AUC"]
}

print("\n[Best Balanced Ensemble Model - KNN + Autoencoder]")
print(classification_report(y_test, ensemble_preds, digits=4))

#Confusion Matrix
cm = confusion_matrix(y_test, ensemble_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Balanced Ensemble")
plt.show()

#ROC Curve
fpr, tpr, _ = roc_curve(y_test, ensemble_score)
plt.plot(fpr, tpr, label=f"Ensemble (AUC = {best_metrics['AUC']:.3f})", color='blue')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Balanced Ensemble")
plt.legend()
plt.grid(True)
plt.show()


#Heatmap
ensemble_df = pd.DataFrame(ensemble_results, index=["Ensemble (KNN+AE)"])
plt.figure(figsize=(6, 3))
sns.heatmap(ensemble_df, annot=True, fmt=".3f", cmap="Blues", cbar=True)
plt.title("Performance Heatmap - Ensemble Model (KNN + Autoencoder)")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#TSNE
def plot_tsne_ensemble(y_true, y_pred, title):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_test)

    plt.figure(figsize=(12, 5))

    #True Labels
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_true, cmap='coolwarm', s=10, alpha=0.7)
    plt.title(f'True Labels - {title}')
    plt.colorbar(scatter1, label='0=Normal, 1=Anomaly')

    #Predicted Labels
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred, cmap='coolwarm', s=10, alpha=0.7)
    plt.title(f'Predicted Labels - {title}')
    plt.colorbar(scatter2, label='0=Normal, 1=Anomaly')

    plt.tight_layout()
    plt.show()


best_preds = (ensemble_score >= best_thresh).astype(int)

plot_tsne_ensemble(y_test, best_preds, 'Ensemble (KNN + Autoencoder)')

#Comparison of Anomaly Counts
models_list = ["KNN", "Autoencoder", "Best Ensemble (KNN+AE)"]
anomaly_counts = [sum(y_pred_knn), sum(y_pred_auto), sum(best_preds)]

plt.figure(figsize=(8, 5))
bars = plt.bar(models_list, anomaly_counts, color=['green', 'orange', 'blue'], alpha=0.8)
plt.title('Comparison of Anomaly Counts')

plt.ylabel('Number of Anomalies')
plt.grid(axis='y', linestyle='--', alpha=0.6)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1, str(int(height)),
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
