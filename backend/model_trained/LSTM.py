import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ======= CẤU HÌNH =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= LOAD DATA =======
train_df = pd.read_csv("../data/DSRL-APT-2023.csv")

# ======= TIỀN XỬ LÝ =======
columns_to_drop = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp", "Activity"]
train_features = train_df.drop(columns=columns_to_drop)

# Convert labels to a more understandable format and then encode them
# train_labels_raw = train_df["Stage"].map({
#     'Benign': 'Không có',
#     'Reconnaissance': 'Rất thấp',
#     'Establish Foothold': 'Thấp',
#     'Lateral Movement': 'Trung bình',
#     'Data Exfiltration': 'Cao'
# })

train_labels_raw = train_df["Stage"].map({
    'Benign': 'Không có',
    'Reconnaissance': 'Thấp',
    'Establish Foothold': 'Trung bình',
    'Lateral Movement': 'Cao',
    'Data Exfiltration': 'Rất cao'
})

# ======= ENCODING =======
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels_raw)

# ======= SCALING =======
scaler = StandardScaler()

# Ensure only numeric data is passed into SMOTE
train_features_numeric = train_features.select_dtypes(include=[np.number])

# Apply scaling to numeric features only
X_train_scaled = scaler.fit_transform(train_features_numeric)

# ======= CHIA DỮ LIỆU =======
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_train_scaled, train_labels, test_size=0.2, stratify=train_labels, random_state=42)

# ======= CÂN BẰNG DỮ LIỆU TRÊN TẬP HUẤN LUYỆN =======
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# ======= CHUYỂN SANG TENSOR =======
X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.long)

# Reshape để phù hợp với LSTM (batch_size, sequence_length, input_size)
X_train_tensor = X_train_tensor.unsqueeze(1)  # Thêm chiều cho sequence_length

# Tạo DataLoader cho tập huấn luyện
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Tạo DataLoader cho tập kiểm tra
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ======= MÔ HÌNH LSTM =======
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=5, num_layers=2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Chỉ lấy đầu ra của timestep cuối cùng
        return output

model = LSTM(X_train_resampled.shape[1]).to(device)

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

criterion = FocalLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ======= TRAINING =======
epochs = 30
best_f1 = 0
patience = 5
patience_counter = 0

print("Bắt đầu huấn luyện...")

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

    # Đánh giá sau mỗi epoch
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in train_loader:
            xb = xb.to(device)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, output_dict=True)
    f1_score = report["macro avg"]["f1-score"]

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, F1-score: {f1_score:.4f}")

    # Early stopping
    if f1_score > best_f1:
        best_f1 = f1_score
        patience_counter = 0
        torch.save(model.state_dict(), "lstm_dsrl.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping due to no improvement")
            break

# ======= EVALUATION =======
model.load_state_dict(torch.load("lstm_dsrl.pth"))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        outputs = model(xb)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("== Báo cáo đánh giá mô hình LSTM sử dụng dataset DSRL APT 2023 ==")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))