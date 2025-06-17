import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import optuna

# ======= CẤU HÌNH =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= LOAD DATA =======
train_df = pd.read_csv("DSRL-APT-2023.csv")

# ======= TIỀN XỬ LÝ =======
columns_to_drop = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp", "Activity"]
train_features = train_df.drop(columns=columns_to_drop)
train_labels_raw = train_df["Stage"]

# ======= ENCODING =======
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels_raw)

# ======= SCALING =======
scaler = StandardScaler()
train_features_numeric = train_features.select_dtypes(include=[np.number])
X_scaled = scaler.fit_transform(train_features_numeric)

# ======= CHIA DỮ LIỆU =======
X_train_scaled, X_temp_scaled, y_train, y_temp = train_test_split(
    X_scaled, train_labels, test_size=0.3, stratify=train_labels, random_state=42
)
X_val_scaled, X_test_scaled, y_val, y_test = train_test_split(
    X_temp_scaled, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# ======= CÂN BẰNG DỮ LIỆU =======
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# ======= PCA =======
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_resampled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# ======= MÔ HÌNH LSTM =======
class APTLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(APTLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

# ======= FOCAL LOSS =======
class FocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return loss

# ======= HÀM ĐÁNH GIÁ =======
def evaluate_model(model, loader, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            outputs = model(xb)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Tính các chỉ số cho từng lớp
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    # Tạo dictionary chứa kết quả
    metrics = {
        'precision': {label_encoder.classes_[i]: precision[i] for i in range(num_classes)},
        'recall': {label_encoder.classes_[i]: recall[i] for i in range(num_classes)},
        'f1': {label_encoder.classes_[i]: f1[i] for i in range(num_classes)}
    }

    # Tạo báo cáo chi tiết
    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0, output_dict=False)

    return metrics, all_labels, all_preds, report

# ======= HÀM TỐI ƯU SIÊU THAM SỐ =======
def objective(trial):
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])

    X_val_tensor = torch.tensor(X_val_pca, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = APTLSTM(
        input_dim=X_train_pca.shape[1],
        hidden_dim=hidden_dim,
        output_dim=len(label_encoder.classes_),
        num_layers=num_layers,
        dropout_rate=dropout_rate
    ).to(device)

    criterion = FocalLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 10
    best_f1 = 0
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        metrics, _, _, _ = evaluate_model(model, val_loader, num_classes=len(label_encoder.classes_))
        avg_f1 = np.mean(list(metrics['f1'].values()))

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_f1

# ======= CHẠY TỐI ƯU =======
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
best_params = study.best_params
print("Best hyperparameters:", best_params)

# ======= HUẤN LUYỆN MÔ HÌNH CUỐI CÙNG =======
X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.long)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)

X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

model = APTLSTM(
    input_dim=X_train_pca.shape[1],
    hidden_dim=best_params['hidden_dim'],
    output_dim=len(label_encoder.classes_),
    num_layers=best_params['num_layers'],
    dropout_rate=best_params['dropout_rate']
).to(device)

criterion = FocalLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])

epochs = 20
best_f1 = 0
patience = 5
patience_counter = 0

print("Bắt đầu huấn luyện mô hình cuối cùng...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    metrics, _, _, report = evaluate_model(model, train_loader, num_classes=len(label_encoder.classes_))
    avg_f1 = np.mean(list(metrics['f1'].values()))

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Avg F1: {avg_f1:.4f}")
    print("\nClassification Report (Train):\n", report)

    if avg_f1 > best_f1:
        best_f1 = avg_f1
        patience_counter = 0
        torch.save(model.state_dict(), "best_apt_lstm.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping due to no improvement")
            break

# ======= ĐÁNH GIÁ CUỐI CÙNG =======
model.load_state_dict(torch.load("best_apt_lstm.pth"))
metrics, _, _, report = evaluate_model(model, test_loader, num_classes=len(label_encoder.classes_))

print("== Kết quả đánh giá trên tập test ==")
print("Precision Scores:")
for label, score in metrics['precision'].items():
    print(f"{label}: {score:.4f}")
print("\nRecall Scores:")
for label, score in metrics['recall'].items():
    print(f"{label}: {score:.4f}")
print("\nF1 Scores:")
for label, score in metrics['f1'].items():
    print(f"{label}: {score:.4f}")
print("\nClassification Report (Test):\n", report)


