from flask import Blueprint, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from models import db, User, Upload, Prediction, RiskSummary
import bcrypt
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
import os
import subprocess
import uuid
import platform
from datetime import timedelta

routes = Blueprint('routes', __name__)

# ====== JWT SETUP ======
jwt = JWTManager()

# ====== LOAD SCALER, LABEL_ENCODER ======
scaler = joblib.load('model_trained/scaler.pkl')
label_encoder = joblib.load('model_trained/label_encoder.pkl')

# ====== MODEL ======
class APTLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=5, num_layers=2):
        super(APTLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = scaler.n_features_in_
model = APTLSTM(input_dim).to(device)
model.load_state_dict(torch.load('model_trained/lstm_dsrl.pth', map_location=device))
model.eval()

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

@routes.route('/login', methods=['POST', 'OPTIONS'])
def login():
    print(f"[Login] Handling {request.method} request for /login")
    if request.method == 'OPTIONS':
        print("[Login] Returning 200 for OPTIONS /login")
        return '', 200

    data = request.get_json()
    print(f"[Login] Request data: {data}")
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        print("[Login] Missing username or password")
        return jsonify({'error': 'Missing username or password'}), 400

    user = User.query.filter_by(username=username).first()
    if not user:
        print(f"[Login] User {username} not found")
        return jsonify({'error': 'User not found'}), 401

    if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
        print(f"[Login] Invalid password for {username}")
        return jsonify({'error': 'Invalid password'}), 401

    access_token = create_access_token(
        identity={'user_id': user.id, 'role': user.role},
        expires_delta=timedelta(days=1)
    )
    print(f"[Login] Login successful for {username}")
    return jsonify({
        'message': 'Login successful',
        'access_token': access_token,
        'user': {'id': user.id, 'username': user.username, 'role': user.role}
    }), 200

@routes.route('/register', methods=['POST', 'OPTIONS'])
def register():
    print(f"[Register] Handling {request.method} request for /register")
    if request.method == 'OPTIONS':
        print("[Register] Returning 200 for OPTIONS /register")
        return '', 200

    data = request.get_json()
    print(f"[Register] Request data: {data}")
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        print("[Register] Missing username, email, or password")
        return jsonify({'error': 'Missing username, email, or password'}), 400

    if User.query.filter_by(username=username).first():
        print(f"[Register] Username {username} already exists")
        return jsonify({'error': 'Username already exists'}), 400
    if User.query.filter_by(email=email).first():
        print(f"[Register] Email {email} already exists")
        return jsonify({'error': 'Email already exists'}), 400

    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    user = User(username=username, email=email, password_hash=password_hash, role='user')
    db.session.add(user)
    db.session.commit()
    print(f"[Register] User {username} registered successfully")
    return jsonify({'message': 'User registered successfully'}), 201

# ====== ROUTE: NHẬN 1 LOG ======
@routes.route('/predict', methods=['POST'])
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
        risk_label = label_encoder.inverse_transform([pred_idx])[0]

    pred = Prediction(upload_id=None, log_data=data['log'], predicted_label=risk_label)
    db.session.add(pred)
    db.session.commit()

    return jsonify({'risk_level': risk_label})

# ====== ROUTE: NHẬN FILE LOG ======
@routes.route('/upload-logs', methods=['POST'])
@jwt_required()
def upload_logs():
    current_user = get_jwt_identity()
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file'}), 400

    file = request.files['file']
    filename = file.filename
    save_path = os.path.join('Uploads', filename)
    os.makedirs('Uploads', exist_ok=True)
    file.save(save_path)

    lines = open(save_path, encoding='utf-8').read().strip().split('\n')
    feature_list = []
    raw_logs = []

    for line in lines:
        if line.strip():
            features = parse_log(line)
            feature_list.append(features)
            raw_logs.append(line)

    if not feature_list:
        return jsonify({'error': 'Invalid log file'}), 400

    upload = Upload(filename=filename, file_path=save_path, user_id=current_user['user_id'])
    db.session.add(upload)
    db.session.commit()

    X = np.vstack(feature_list)
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        risk_labels = label_encoder.inverse_transform(preds)

    risk_count = {}
    for log_data, risk_label in zip(raw_logs, risk_labels):
        pred = Prediction(upload_id=upload.id, log_data=log_data, predicted_label=risk_label)
        db.session.add(pred)
        risk_count[risk_label] = risk_count.get(risk_label, 0) + 1

    for risk_level, count in risk_count.items():
        summary = RiskSummary(upload_id=upload.id, risk_level=risk_level, count=count)
        db.session.add(summary)

    db.session.commit()

    results = [{'log_index': i, 'risk_level': label} for i, label in enumerate(risk_labels)]
    return jsonify({'upload_id': upload.id, 'results': results})

# ====== ROUTE: NHẬN FILE PCAP ======
@routes.route('/upload-pcap', methods=['POST'])
@jwt_required()
def upload_pcap():
    current_user = get_jwt_identity()
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file'}), 400

    file = request.files['file']
    if not file.filename.lower().endswith('.pcap'):
        return jsonify({'error': 'Invalid PCAP file'}), 400

    pcap_filename = file.filename
    pcap_save_path = os.path.join('Uploads', pcap_filename)
    os.makedirs('Uploads', exist_ok=True)
    file.save(pcap_save_path)

    csv_filename = f"{uuid.uuid4()}.csv"
    csv_save_path = os.path.join('Uploads', csv_filename)

    cicflowmeter_dir = os.path.join(os.path.dirname(__file__), 'cicflowmeter')
    venv_activate = os.path.join(cicflowmeter_dir, '.venv', 'bin', 'activate') if platform.system() != 'Windows' else os.path.join(cicflowmeter_dir, '.venv', 'Scripts', 'activate.bat')

    if platform.system() == 'Windows':
        cmd = f'"{venv_activate}" && cicflowmeter -f "{pcap_save_path}" -c "{csv_save_path}"'
    else:
        cmd = f'. "{venv_activate}" && cicflowmeter -f "{pcap_save_path}" -c "{csv_save_path}"'

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return jsonify({'error': 'CICFlowMeter failed'}), 500
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'CICFlowMeter timed out'}), 500

    df = pd.read_csv(csv_save_path)
    df['Flow ID'] = 0
    df = df.rename(columns=column_mapping)
    new_df = pd.DataFrame({col: df.get(col, 0) for col in desired_columns})

    refactored_csv_path = os.path.join('Uploads', f"refactored_{csv_filename}")
    new_df.to_csv(refactored_csv_path, index=False, header=False)

    feature_list = [parse_csv_row(row) for _, row in new_df.iterrows()]
    raw_logs = [row.to_json() for _, row in new_df.iterrows()]

    if not feature_list:
        return jsonify({'error': 'Invalid CSV data'}), 400

    upload = Upload(filename=pcap_filename, file_path=pcap_save_path, user_id=current_user['user_id'])
    db.session.add(upload)
    db.session.commit()

    X = np.vstack(feature_list)
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        risk_labels = label_encoder.inverse_transform(preds)

    risk_count = {}
    for log_data, risk_label in zip(raw_logs, risk_labels):
        pred = Prediction(upload_id=upload.id, log_data=log_data, predicted_label=risk_label)
        db.session.add(pred)
        risk_count[risk_label] = risk_count.get(risk_label, 0) + 1

    for risk_level, count in risk_count.items():
        summary = RiskSummary(upload_id=upload.id, risk_level=risk_level, count=count)
        db.session.add(summary)

    db.session.commit()

    for path in [pcap_save_path, csv_save_path, refactored_csv_path]:
        if os.path.exists(path):
            os.remove(path)

    results = [{'log_index': i, 'risk_level': label} for i, label in enumerate(risk_labels)]
    return jsonify({'upload_id': upload.id, 'results': results})

# ====== ROUTE LỊCH SỬ LOG ======
@routes.route('/upload-history', methods=['GET'])
@jwt_required()
def upload_history():
    current_user = get_jwt_identity()
    user_id = current_user['user_id']
    role = current_user['role']

    if role == 'admin':
        uploads = Upload.query.order_by(Upload.upload_time.desc()).all()
    else:
        uploads = Upload.query.filter_by(user_id=user_id).order_by(Upload.upload_time.desc()).all()

    results = [{
        'upload_id': u.id,
        'filename': u.filename,
        'file_path': u.file_path,
        'upload_time': u.upload_time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_logs': Prediction.query.filter_by(upload_id=u.id).count()
    } for u in uploads]
    return jsonify(results)

# ====== ROUTE LẤY CHI TIẾT 1 LẦN UPLOAD ======
@routes.route('/upload-details/<int:upload_id>', methods=['GET'])
@jwt_required()
def upload_details(upload_id):
    current_user = get_jwt_identity()
    user_id = current_user['user_id']
    role = current_user['role']

    upload = Upload.query.get_or_404(upload_id)
    if role != 'admin' and upload.user_id != user_id:
        return jsonify({'error': 'Unauthorized access'}), 403

    predictions = Prediction.query.filter_by(upload_id=upload_id).all()
    results = [{
        'id': p.id,
        'log_data': p.log_data,
        'predicted_label': p.predicted_label,
        'created_at': p.created_at.strftime('%Y-%m-%d %H:%M:%S')
    } for p in predictions]
    return jsonify(results)

# ====== ROUTE THỐNG KÊ RỦI RO ======
@routes.route('/risk-stats', methods=['GET'])
@jwt_required()
def risk_stats():
    current_user = get_jwt_identity()
    user_id = current_user['user_id']
    role = current_user['role']

    if role == 'admin':
        total_files = Upload.query.count()
        total_logs = Prediction.query.count()
        summary = db.session.query(
            RiskSummary.risk_level,
            db.func.sum(RiskSummary.count)
        ).group_by(RiskSummary.risk_level).all()
    else:
        total_files = Upload.query.filter_by(user_id=user_id).count()
        total_logs = Prediction.query.join(Upload).filter(Upload.user_id == user_id).count()
        summary = db.session.query(
            RiskSummary.risk_level,
            db.func.sum(RiskSummary.count)
        ).join(Upload).filter(Upload.user_id == user_id).group_by(RiskSummary.risk_level).all()

    risk_overview = [{'risk_level': r[0], 'count': r[1]} for r in summary]
    return jsonify({
        'total_files': total_files,
        'total_logs': total_logs,
        'risk_overview': risk_overview
    })

# ====== ROUTE LẤY DANH SÁCH UPLOADS ======
@routes.route('/Uploads', methods=['GET'])
@jwt_required()
def get_uploads():
    current_user = get_jwt_identity()
    user_id = current_user['user_id']
    role = current_user['role']

    if role == 'admin':
        uploads = Upload.query.order_by(Upload.upload_time.desc()).all()
    else:
        uploads = Upload.query.filter_by(user_id=user_id).order_by(Upload.upload_time.desc()).all()

    data = [{
        'upload_id': u.id,
        'filename': u.filename,
        'upload_time': u.upload_time.strftime('%Y-%m-%d %H:%M'),
        'total_logs': Prediction.query.filter_by(upload_id=u.id).count(),
        'risk_summary': [{'risk_level': rs.risk_level, 'count': rs.count} for rs in u.risk_summaries]
    } for u in uploads]
    return jsonify(data)

# ====== ROUTE LẤY THÔNG TIN PROFILE ======
@routes.route('/profile', methods=['GET', 'OPTIONS'])
@jwt_required()
def get_profile():
    print(f"[Profile] Handling {request.method} request for /profile")
    if request.method == 'OPTIONS':
        print("[Profile] Returning 200 for OPTIONS /profile")
        return '', 200

    try:
        current_user = get_jwt_identity()
        user = User.query.get_or_404(current_user['user_id'])
        print(f"[Profile] Fetched profile for user {user.username}")
        return jsonify({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role,
            'created_at': user.created_at.isoformat()
        }), 200
    except Exception as e:
        print(f"[Profile] Error: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# ====== ROUTE CẬP NHẬT PROFILE ======
@routes.route('/profile', methods=['PUT', 'OPTIONS'])
@jwt_required()
def update_profile():
    print(f"[Profile] Handling {request.method} request for /profile")
    if request.method == 'OPTIONS':
        print("[Profile] Returning 200 for OPTIONS /profile")
        return '', 200

    try:
        current_user = get_jwt_identity()
        user = User.query.get_or_404(current_user['user_id'])
        data = request.get_json()
        print(f"[Profile] Update data: {data}")

        username = data.get('username')
        email = data.get('email')

        if not username or not email:
            print("[Profile] Missing username or email")
            return jsonify({'error': 'Missing username or email'}), 400

        if username != user.username and User.query.filter_by(username=username).first():
            print(f"[Profile] Username {username} already exists")
            return jsonify({'error': 'Username already exists'}), 400
        if email != user.email and User.query.filter_by(email=email).first():
            print(f"[Profile] Email {email} already exists")
            return jsonify({'error': 'Email already exists'}), 400

        user.username = username
        user.email = email
        db.session.commit()
        print(f"[Profile] Updated profile for user {user.username}")
        return jsonify({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role,
            'created_at': user.created_at.isoformat()
        }), 200
    except Exception as e:
        print(f"[Profile] Error: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# ====== ROUTE ĐỔI MẬT KHẨU ======
@routes.route('/change-password', methods=['POST', 'OPTIONS'])
@jwt_required()
def change_password():
    print(f"[Change-Password] Handling {request.method} request for /change-password")
    if request.method == 'OPTIONS':
        print("[Change-Password] Returning 200 for OPTIONS /change-password")
        return '', 200

    try:
        current_user = get_jwt_identity()
        user = User.query.get_or_404(current_user['user_id'])
        data = request.get_json()
        print(f"[Change-Password] Request data: {data}")

        current_password = data.get('currentPassword')
        new_password = data.get('newPassword')

        if not current_password or not new_password:
            print("[Change-Password] Missing current or new password")
            return jsonify({'error': 'Missing current or new password'}), 400

        if not bcrypt.checkpw(current_password.encode('utf-8'), user.password_hash.encode('utf-8')):
            print("[Change-Password] Invalid current password")
            return jsonify({'error': 'Invalid current password'}), 401

        user.password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        db.session.commit()
        print(f"[Change-Password] Password changed for user {user.username}")
        return jsonify({'message': 'Password changed successfully'}), 200
    except Exception as e:
        print(f"[Change-Password] Error: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# ====== ROUTE LẤY DANH SÁCH NGƯỜI DÙNG ======
@routes.route('/users', methods=['GET', 'OPTIONS'])
@jwt_required()
def get_users():
    print(f"[Users] Handling {request.method} request for /users")
    if request.method == 'OPTIONS':
        print("[Users] Returning 200 for OPTIONS /users")
        return '', 200

    try:
        current_user = get_jwt_identity()
        if current_user['role'] != 'admin':
            print("[Users] Unauthorized access attempt")
            return jsonify({'error': 'Admin access required'}), 403

        users = User.query.filter_by(role='user').all()  # Chỉ lấy user, không lấy admin
        data = [{
            'id': u.id,
            'username': u.username,
            'email': u.email,
            'role': u.role,
            'created_at': u.created_at.isoformat()
        } for u in users]
        print(f"[Users] Fetched {len(data)} users")
        return jsonify(data), 200
    except Exception as e:
        print(f"[Users] Error: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# ====== ROUTE LẤY THÔNG TIN MỘT NGƯỜI DÙNG ======
@routes.route('/users/<int:user_id>', methods=['GET', 'OPTIONS'])
@jwt_required()
def get_user(user_id):
    print(f"[User] Handling {request.method} request for /users/{user_id}")
    if request.method == 'OPTIONS':
        print("[User] Returning 200 for OPTIONS /users/{user_id}")
        return '', 200

    try:
        current_user = get_jwt_identity()
        if current_user['role'] != 'admin':
            print("[User] Unauthorized access attempt")
            return jsonify({'error': 'Admin access required'}), 403

        user = User.query.get_or_404(user_id)
        print(f"[User] Fetched user {user.username}")
        return jsonify({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role,
            'created_at': user.created_at.isoformat()
        }), 200
    except Exception as e:
        print(f"[User] Error: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# ====== ROUTE CẬP NHẬT NGƯỜI DÙNG ======
@routes.route('/users/<int:user_id>', methods=['PUT', 'OPTIONS'])
@jwt_required()
def update_user(user_id):
    print(f"[User] Handling {request.method} request for /users/{user_id}")
    if request.method == 'OPTIONS':
        print("[User] Returning 200 for OPTIONS /users/{user_id}")
        return '', 200

    try:
        current_user = get_jwt_identity()
        if current_user['role'] != 'admin':
            print("[User] Unauthorized access attempt")
            return jsonify({'error': 'Admin access required'}), 403

        user = User.query.get_or_404(user_id)
        data = request.get_json()
        print(f"[User] Update data for user {user_id}: {data}")

        username = data.get('username')
        email = data.get('email')
        new_password = data.get('newPassword')

        if not username or not email:
            print("[User] Missing username or email")
            return jsonify({'error': 'Missing username or email'}), 400

        if username != user.username and User.query.filter_by(username=username).first():
            print(f"[User] Username {username} already exists")
            return jsonify({'error': 'Username already exists'}), 400
        if email != user.email and User.query.filter_by(email=email).first():
            print(f"[User] Email {email} already exists")
            return jsonify({'error': 'Email already exists'}), 400

        user.username = username
        user.email = email
        if new_password:
            user.password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        db.session.commit()
        print(f"[User] Updated user {user.username}")
        return jsonify({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'role': user.role,
            'created_at': user.created_at.isoformat()
        }), 200
    except Exception as e:
        print(f"[User] Error: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# ====== ROUTE XÓA NGƯỜI DÙNG ======
@routes.route('/users/<int:user_id>', methods=['DELETE', 'OPTIONS'])
@jwt_required()
def delete_user(user_id):
    print(f"[User] Handling {request.method} request for /users/{user_id}")
    if request.method == 'OPTIONS':
        print("[User] Returning 200 for OPTIONS /users/{user_id}")
        return '', 200

    try:
        current_user = get_jwt_identity()
        if current_user['role'] != 'admin':
            print("[User] Unauthorized access attempt")
            return jsonify({'error': 'Admin access required'}), 403
        if current_user['user_id'] == user_id:
            print("[User] Cannot delete self")
            return jsonify({'error': 'Cannot delete your own account'}), 400

        user = User.query.get_or_404(user_id)
        db.session.delete(user)
        db.session.commit()
        print(f"[User] Deleted user {user_id}")
        return jsonify({'message': 'User deleted successfully'}), 200
    except Exception as e:
        print(f"[User] Error: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500