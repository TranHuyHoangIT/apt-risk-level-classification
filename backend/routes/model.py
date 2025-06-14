from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import db, Prediction
import torch
import torch.nn as nn
import numpy as np
import joblib
# import pandas as pd

model_bp = Blueprint('model', __name__)

# ====== LOAD SCALER, LABEL_ENCODER ======
scaler = joblib.load('model_trained/scaler.pkl')
label_encoder = joblib.load('model_trained/label_encoder.pkl')

# ====== MODEL ======
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=5, num_layers=2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = scaler.n_features_in_
model = LSTM(input_dim).to(device)
model.load_state_dict(torch.load('model_trained/lstm_dsrl.pth', map_location=device))
model.eval()

__all__ = ['model', 'scaler', 'label_encoder', 'parse_log', 'parse_csv_row', 'desired_columns', 'column_mapping', 'device']

# ====== COLUMN MAPPING FOR CICFlowMeter CSV ======
desired_columns = [
    'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp',
    'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets', 'Total Length of Fwd Packet',
    'Total Length of Bwd Packet', 'Fwd Packet Length Max', 'Fwd Packet Length Min',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
    'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
    'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
    'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Packet Length Min', 'Packet Length Max', 'Packet Length Mean', 'Packet Length Std',
    'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWR Flag Count',
    'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Fwd Segment Size Avg',
    'Bwd Segment Size Avg', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg',
    'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'FWD Init Win Bytes', 'Bwd Init Win Bytes', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std',
    'Idle Max', 'Idle Min'
]

column_mapping = {
    'src_ip': 'Src IP',
    'dst_ip': 'Dst IP',
    'src_port': 'Src Port',
    'dst_port': 'Dst Port',
    'protocol': 'Protocol',
    'timestamp': 'Timestamp',
    'flow_duration': 'Flow Duration',
    'tot_fwd_pkts': 'Total Fwd Packet',
    'tot_bwd_pkts': 'Total Bwd packets',
    'totlen_fwd_pkts': 'Total Length of Fwd Packet',
    'totlen_bwd_pkts': 'Total Length of Bwd Packet',
    'fwd_pkt_len_max': 'Fwd Packet Length Max',
    'fwd_pkt_len_min': 'Fwd Packet Length Min',
    'fwd_pkt_len_mean': 'Fwd Packet Length Mean',
    'fwd_pkt_len_std': 'Fwd Packet Length Std',
    'bwd_pkt_len_max': 'Bwd Packet Length Max',
    'bwd_pkt_len_min': 'Bwd Packet Length Min',
    'bwd_pkt_len_mean': 'Bwd Packet Length Mean',
    'bwd_pkt_len_std': 'Bwd Packet Length Std',
    'flow_byts_s': 'Flow Bytes/s',
    'flow_pkts_s': 'Flow Packets/s',
    'flow_iat_mean': 'Flow IAT Mean',
    'flow_iat_std': 'Flow IAT Std',
    'flow_iat_max': 'Flow IAT Max',
    'flow_iat_min': 'Flow IAT Min',
    'fwd_iat_tot': 'Fwd IAT Total',
    'fwd_iat_mean': 'Fwd IAT Mean',
    'fwd_iat_std': 'Fwd IAT Std',
    'fwd_iat_max': 'Fwd IAT Max',
    'fwd_iat_min': 'Fwd IAT Min',
    'bwd_iat_tot': 'Bwd IAT Total',
    'bwd_iat_mean': 'Bwd IAT Mean',
    'bwd_iat_std': 'Bwd IAT Std',
    'bwd_iat_max': 'Bwd IAT Max',
    'bwd_iat_min': 'Bwd IAT Min',
    'fwd_psh_flags': 'Fwd PSH Flags',
    'bwd_psh_flags': 'Bwd PSH Flags',
    'fwd_urg_flags': 'Fwd URG Flags',
    'bwd_urg_flags': 'Bwd URG Flags',
    'fwd_header_len': 'Fwd Header Length',
    'bwd_header_len': 'Bwd Header Length',
    'fwd_pkts_s': 'Fwd Packets/s',
    'bwd_pkts_s': 'Bwd Packets/s',
    'pkt_len_min': 'Packet Length Min',
    'pkt_len_max': 'Packet Length Max',
    'pkt_len_mean': 'Packet Length Mean',
    'pkt_len_std': 'Packet Length Std',
    'pkt_len_var': 'Packet Length Variance',
    'fin_flag_cnt': 'FIN Flag Count',
    'syn_flag_cnt': 'SYN Flag Count',
    'rst_flag_cnt': 'RST Flag Count',
    'psh_flag_cnt': 'PSH Flag Count',
    'ack_flag_cnt': 'ACK Flag Count',
    'urg_flag_cnt': 'URG Flag Count',
    'cwr_flag_count': 'CWR Flag Count',
    'ece_flag_cnt': 'ECE Flag Count',
    'down_up_ratio': 'Down/Up Ratio',
    'pkt_size_avg': 'Average Packet Size',
    'fwd_seg_size_avg': 'Fwd Segment Size Avg',
    'bwd_seg_size_avg': 'Bwd Segment Size Avg',
    'fwd_byts_b_avg': 'Fwd Bytes/Bulk Avg',
    'fwd_pkts_b_avg': 'Fwd Packet/Bulk Avg',
    'fwd_blk_rate_avg': 'Fwd Bulk Rate Avg',
    'bwd_byts_b_avg': 'Bwd Bytes/Bulk Avg',
    'bwd_pkts_b_avg': 'Bwd Packet/Bulk Avg',
    'bwd_blk_rate_avg': 'Bwd Bulk Rate Avg',
    'subflow_fwd_pkts': 'Subflow Fwd Packets',
    'subflow_fwd_byts': 'Subflow Fwd Bytes',
    'subflow_bwd_pkts': 'Subflow Bwd Packets',
    'subflow_bwd_byts': 'Subflow Bwd Bytes',
    'init_fwd_win_byts': 'FWD Init Win Bytes',
    'init_bwd_win_byts': 'Bwd Init Win Bytes',
    'fwd_act_data_pkts': 'Fwd Act Data Pkts',
    'fwd_seg_size_min': 'Fwd Seg Size Min',
    'active_mean': 'Active Mean',
    'active_std': 'Active Std',
    'active_max': 'Active Max',
    'active_min': 'Active Min',
    'idle_mean': 'Idle Mean',
    'idle_std': 'Idle Std',
    'idle_max': 'Idle Max',
    'idle_min': 'Idle Min'
}

# ====== PARSE LOG ======
def parse_log(log_str):
    parts = log_str.strip().split(',')
    if len(parts) < 8:
        raise ValueError("Log không đủ cột")
    features = [float(x) if x != '' else 0.0 for x in parts[7:]]
    return np.array(features)

# ====== PARSE CSV ROW ======
def parse_csv_row(row):
    metadata_columns = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp"]
    feature_columns = [col for col in desired_columns if col not in metadata_columns]
    features = []
    for col in feature_columns:
        try:
            value = float(row.get(col, 0.0)) if row.get(col, '') != '' else 0.0
        except (ValueError, TypeError):
            value = 0.0
        features.append(value)
    return np.array(features)

@model_bp.route('/predict', methods=['POST'])
@jwt_required()
def predict():
    current_user = get_jwt_identity()
    data = request.get_json()
    if not data or 'log' not in data:
        return jsonify({'error': 'Missing log'}), 400

    X = parse_log(data['log']).reshape(1, -1)
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        pred_idx = torch.argmax(outputs, dim=1).cpu().item()
        stage_label = label_encoder.inverse_transform([pred_idx])[0]

    pred = Prediction(upload_id=None, log_data=data['log'], predicted_label=stage_label)
    db.session.add(pred)
    db.session.commit()

    return jsonify({'stage_label': stage_label})