from flask import Blueprint, request, jsonify
from models import db, Upload, Prediction, RiskSummary
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
import os

routes = Blueprint('routes', __name__)

# ====== LOAD SCALER, LABEL_ENCODER ======
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# ====== MODEL ======
class APTLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=5, num_layers=2):
        super(APTLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Chỉ lấy đầu ra của timestep cuối cùng
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = scaler.n_features_in_  # Số chiều đầu vào từ scaler
model = APTLSTM(input_dim).to(device)
model.load_state_dict(torch.load('lstm_dsrl.pth', map_location=device))
model.eval()

# ====== PARSE LOG ======
def parse_log(log_str):
    parts = log_str.strip().split(',')
    features = parts[7:]  # Giả định các đặc trưng bắt đầu từ cột thứ 8
    features = [float(x) if x != '' else 0.0 for x in features]
    return np.array(features)

# ====== ROUTE: NHẬN 1 LOG ======
@routes.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'log' not in data:
            return jsonify({'error': 'Bạn phải gửi log dạng JSON: {"log": "..."}'}), 400

        log_str = data['log']
        X = parse_log(log_str).reshape(1, -1)
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(device)  # Thêm chiều sequence_length

        with torch.no_grad():
            outputs = model(X_tensor)
            pred_idx = torch.argmax(outputs, dim=1).cpu().item()
            risk_label = label_encoder.inverse_transform([pred_idx])[0]

        # Lưu Prediction vào DB (upload_id = None vì không qua file)
        pred = Prediction(upload_id=None, log_data=log_str, predicted_label=risk_label)
        db.session.add(pred)
        db.session.commit()

        return jsonify({'risk_level': risk_label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ====== ROUTE: NHẬN FILE LOG ======
@routes.route('/upload-logs', methods=['POST'])
def upload_logs():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Bạn phải upload file log qua form-data key="file"'}), 400

        file = request.files['file']
        filename = file.filename
        save_path = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
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
            return jsonify({'error': 'File không chứa log hợp lệ'}), 400

        # Tạo Upload record
        upload = Upload(filename=filename, file_path=save_path)
        db.session.add(upload)
        db.session.commit()

        X = np.vstack(feature_list)
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(device)  # Thêm chiều sequence_length

        with torch.no_grad():
            outputs = model(X_tensor)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            risk_labels = label_encoder.inverse_transform(preds)

        # Lưu từng log + prediction
        risk_count = {}
        for log_data, risk_label in zip(raw_logs, risk_labels):
            pred = Prediction(upload_id=upload.id, log_data=log_data, predicted_label=risk_label)
            db.session.add(pred)

            if risk_label not in risk_count:
                risk_count[risk_label] = 0
            risk_count[risk_label] += 1

        # Lưu risk_summary
        for risk_level, count in risk_count.items():
            summary = RiskSummary(upload_id=upload.id, risk_level=risk_level, count=count)
            db.session.add(summary)

        db.session.commit()

        # Trả kết quả
        results = [{'log_index': i, 'risk_level': label} for i, label in enumerate(risk_labels)]
        return jsonify({'upload_id': upload.id, 'results': results})

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# ====== ROUTE LỊCH SỬ LOG ======
@routes.route('/upload-history', methods=['GET'])
def upload_history():
    try:
        uploads = Upload.query.order_by(Upload.upload_time.desc()).all()
        results = []
        for u in uploads:
            total_logs = Prediction.query.filter_by(upload_id=u.id).count()
            results.append({
                'upload_id': u.id,
                'filename': u.filename,
                'file_path': u.file_path,
                'upload_time': u.upload_time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_logs': total_logs
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ====== ROUTE LẤY CHI TIẾT 1 LẦN UPLOAD ======
@routes.route('/upload-details/<int:upload_id>', methods=['GET'])
def upload_details(upload_id):
    try:
        predictions = Prediction.query.filter_by(upload_id=upload_id).all()
        results = []
        for p in predictions:
            results.append({
                'id': p.id,
                'log_data': p.log_data,
                'predicted_label': p.predicted_label,
                'created_at': p.created_at.strftime('%Y-%m-%d %H:%M:%S')
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ====== ROUTE THỐNG KÊ RỦI RO ======
@routes.route('/risk-stats', methods=['GET'])
def risk_stats():
    total_files = Upload.query.count()
    total_logs = Prediction.query.count()

    # Group by risk_level, sum count
    summary = db.session.query(
        RiskSummary.risk_level,
        db.func.sum(RiskSummary.count)
    ).group_by(RiskSummary.risk_level).all()

    risk_overview = [{'risk_level': r[0], 'count': r[1]} for r in summary]

    return jsonify({
        'total_files': total_files,
        'total_logs': total_logs,
        'risk_overview': risk_overview
    })

# ====== ROUTE LẤY DANH SÁCH UPLOADS ======
@routes.route('/uploads', methods=['GET'])
def get_uploads():
    uploads = Upload.query.order_by(Upload.upload_time.desc()).all()
    data = []
    for u in uploads:
        total_logs = Prediction.query.filter_by(upload_id=u.id).count()
        risk_summary = [
            {'risk_level': rs.risk_level, 'count': rs.count}
            for rs in u.risk_summaries
        ]
        data.append({
            'upload_id': u.id,
            'filename': u.filename,
            'upload_time': u.upload_time.strftime('%Y-%m-%d %H:%M'),
            'total_logs': total_logs,
            'risk_summary': risk_summary
        })
    return jsonify(data)