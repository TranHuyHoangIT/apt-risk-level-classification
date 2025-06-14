from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import db, Upload, Prediction, StageSummary
import os
import subprocess
import uuid
import platform
import pandas as pd
import numpy as np
import torch
from .model import model, scaler, label_encoder, parse_log, parse_csv_row, desired_columns, column_mapping, device

upload = Blueprint('upload', __name__)

@upload.route('/upload-logs', methods=['POST'])
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

    upload_entry = Upload(filename=filename, file_path=save_path, user_id=current_user['user_id'])
    db.session.add(upload_entry)
    db.session.commit()

    X = np.vstack(feature_list)
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        stage_labels = label_encoder.inverse_transform(preds)

    stage_count = {}
    for log_data, stage_label in zip(raw_logs, stage_labels):
        pred = Prediction(upload_id=upload_entry.id, log_data=log_data, predicted_label=stage_label)
        db.session.add(pred)
        stage_count[stage_label] = stage_count.get(stage_label, 0) + 1

    for stage_label, count in stage_count.items():
        summary = StageSummary(upload_id=upload_entry.id, stage_label=stage_label, count=count)
        db.session.add(summary)

    db.session.commit()

    results = [{'log_index': i, 'stage_label': label} for i, label in enumerate(stage_labels)]
    return jsonify({'upload_id': upload_entry.id, 'results': results})

@upload.route('/upload-pcap', methods=['POST'])
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

    upload_entry = Upload(filename=pcap_filename, file_path=pcap_save_path, user_id=current_user['user_id'])
    db.session.add(upload_entry)
    db.session.commit()

    X = np.vstack(feature_list)
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        stage_labels = label_encoder.inverse_transform(preds)

    stage_count = {}
    for log_data, stage_label in zip(raw_logs, stage_labels):
        pred = Prediction(upload_id=upload_entry.id, log_data=log_data, predicted_label=stage_label)
        db.session.add(pred)
        stage_count[stage_label] = stage_count.get(stage_label, 0) + 1

    for stage_label, count in stage_count.items():
        summary = StageSummary(upload_id=upload_entry.id, stage_label=stage_label, count=count)
        db.session.add(summary)

    db.session.commit()

    for path in [pcap_save_path, csv_save_path, refactored_csv_path]:
        if os.path.exists(path):
            os.remove(path)

    results = [{'log_index': i, 'stage_label': label} for i, label in enumerate(stage_labels)]
    return jsonify({'upload_id': upload_entry.id, 'results': results})

@upload.route('/upload-history', methods=['GET'])
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

@upload.route('/upload-details/<int:upload_id>', methods=['GET'])
@jwt_required()
def upload_details(upload_id):
    current_user = get_jwt_identity()
    user_id = current_user['user_id']
    role = current_user['role']

    upload_entry = Upload.query.get_or_404(upload_id)
    if role != 'admin' and upload_entry.user_id != user_id:
        return jsonify({'error': 'Unauthorized access'}), 403

    predictions = Prediction.query.filter_by(upload_id=upload_id).all()
    results = [{
        'id': p.id,
        'log_data': p.log_data,
        'predicted_label': p.predicted_label,
        'created_at': p.created_at.strftime('%Y-%m-%d %H:%M:%S')
    } for p in predictions]
    return jsonify(results)

@upload.route('/stage-stats', methods=['GET'])
@jwt_required()
def stage_stats():
    current_user = get_jwt_identity()
    user_id = current_user['user_id']
    role = current_user['role']

    if role == 'admin':
        total_files = Upload.query.count()
        total_logs = Prediction.query.count()
        summary = db.session.query(
            StageSummary.stage_label,
            db.func.sum(StageSummary.count)
        ).group_by(StageSummary.stage_label).all()
    else:
        total_files = Upload.query.filter_by(user_id=user_id).count()
        total_logs = Prediction.query.join(Upload).filter(Upload.user_id == user_id).count()
        summary = db.session.query(
            StageSummary.stage_label,
            db.func.sum(StageSummary.count)
        ).join(Upload).filter(Upload.user_id == user_id).group_by(StageSummary.stage_label).all()

    stage_overview = [{'stage_label': r[0], 'count': r[1]} for r in summary]
    return jsonify({
        'total_files': total_files,
        'total_logs': total_logs,
        'stage_overview': stage_overview
    })

@upload.route('/Uploads', methods=['GET'])
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
        'stage_summary': [{'stage_label': rs.stage_label, 'count': rs.count} for rs in u.stage_summaries]
    } for u in uploads]
    return jsonify(data)