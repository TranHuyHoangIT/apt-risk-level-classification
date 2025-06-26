import os
import uuid
import subprocess
import platform
import pandas as pd
import numpy as np
import torch
import json
import time
from models import db, Upload, Prediction, StageSummary
from flask import Blueprint, request, jsonify, Response
from flask_jwt_extended import jwt_required, get_jwt_identity, verify_jwt_in_request
from collections import deque
from .model import model, scaler, label_encoder, pca, parse_log, parse_csv_row, desired_columns, column_mapping, device
from .auth import get_current_user

upload = Blueprint('upload', __name__)


@upload.route('/api/upload-logs', methods=['POST'])
@jwt_required()
def upload_logs():
    current_user = get_current_user()
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

    # Chuyển feature_list thành DataFrame với tên cột từ scaler
    X = np.vstack(feature_list)
    feature_columns = scaler.feature_names_in_  # Lấy tên cột từ scaler
    X_df = pd.DataFrame(X, columns=feature_columns)

    # Chuẩn hóa và áp dụng PCA
    X_scaled = scaler.transform(X_df)
    X_pca = pca.transform(X_scaled)
    X_tensor = torch.tensor(X_pca, dtype=torch.float32).unsqueeze(1).to(device)

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


@upload.route('/api/upload-pcap', methods=['POST'])
@jwt_required()
def upload_pcap():
    current_user = get_current_user()
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

    cicflowmeter_dir = os.path.join(os.path.dirname(__file__), '..', 'cicflowmeter')
    venv_activate = os.path.join(cicflowmeter_dir, '.venv', 'bin', 'activate') if platform.system() != 'Windows' else os.path.join(cicflowmeter_dir, '.venv', 'Scripts', 'activate.bat')

    if platform.system() == 'Windows':
        cmd = f'"{venv_activate}" && cicflowmeter -f "{pcap_save_path}" -c "{csv_save_path}"'
    else:
        cmd = f'. "{venv_activate}" && cicflowmeter -f "{pcap_save_path}" -c "{csv_save_path}"'

    print("cmd là: ", cmd)

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
    # raw_logs = [row.to_json(orient='values', force_ascii=False) for _, row in new_df.iterrows()]
    raw_logs = [','.join(map(str, row)) for _, row in new_df.iterrows()]

    if not feature_list:
        return jsonify({'error': 'Invalid CSV data'}), 400

    upload_entry = Upload(filename=pcap_filename, file_path=pcap_save_path, user_id=current_user['user_id'])
    db.session.add(upload_entry)
    db.session.commit()

    # Chuyển feature_list thành DataFrame với tên cột từ scaler
    X = np.vstack(feature_list)
    feature_columns = scaler.feature_names_in_
    X_df = pd.DataFrame(X, columns=feature_columns)

    # Chuẩn hóa và áp dụng PCA
    X_scaled = scaler.transform(X_df)
    X_pca = pca.transform(X_scaled)
    X_tensor = torch.tensor(X_pca, dtype=torch.float32).unsqueeze(1).to(device)

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

    results = [{'log_index': i, 'stage_label': label, 'log_data': log} for i, (label, log) in enumerate(zip(stage_labels, raw_logs))]
    return jsonify({'upload_id': upload_entry.id, 'results': results})


@upload.route('/api/upload-history', methods=['GET'])
@jwt_required()
def upload_history():
    current_user = get_current_user()
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


@upload.route('/api/upload-details/<int:upload_id>', methods=['GET'])
@jwt_required()
def upload_details(upload_id):
    current_user = get_current_user()
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


@upload.route('/api/stage-stats', methods=['GET'])
@jwt_required()
def stage_stats():
    current_user = get_current_user()
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


@upload.route('/api/Uploads', methods=['GET'])
@jwt_required()
def get_uploads():
    current_user = get_current_user()
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


user_file_queues = {}
user_processing_status = {}


@upload.route('/api/simulate', methods=['POST'])
@jwt_required()
def simulate():
    current_user = get_current_user()
    print(f"[Backend] Current user: {current_user}")

    # Extract a hashable user identifier
    if isinstance(current_user, dict):
        user_id = current_user.get('user_id')
        if not user_id:
            print("[Backend] Error: No valid user identifier in JWT payload")
            return jsonify({'error': 'Invalid user identifier in JWT payload'}), 400
    else:
        user_id = current_user

    print(f"[Backend] Using user_id: {user_id}")

    if 'file' not in request.files:
        print("[Backend] Error: No file part in request")
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        print("[Backend] Error: No selected file or empty filename")
        return jsonify({'error': 'No selected file or empty filename'}), 400

    filename = file.filename
    print(f"[Backend] Received file: {filename}")
    if not (filename.lower().endswith('.csv') or filename.lower().endswith('.pcap')):
        print(f"[Backend] Error: Invalid file format for {filename}")
        return jsonify({'error': 'Invalid file format. Upload CSV or PCAP'}), 400

    # Initialize user queue if not exists
    if user_id not in user_file_queues:
        user_file_queues[user_id] = deque()
        user_processing_status[user_id] = False

    # Save the uploaded file
    save_path = os.path.join('Uploads', f"{uuid.uuid4()}_{filename}")
    os.makedirs('Uploads', exist_ok=True)
    file.save(save_path)
    user_file_queues[user_id].append({'path': save_path, 'original_name': filename})
    print(f"[Backend] Saved file to: {save_path}")

    # Send immediate response that file was queued
    def generate():
        try:
            # Send file queued confirmation
            yield f"data: {json.dumps({'status': 'file_queued', 'filename': filename, 'queue_length': len(user_file_queues[user_id])})}\n\n"

            # If not already processing, start processing
            if not user_processing_status.get(user_id, False):
                user_processing_status[user_id] = True
                yield from process_user_queue(user_id)
            else:
                print(f"[Backend] User {user_id} already processing, file added to queue")

        except Exception as e:
            print(f"[Backend] General error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            user_processing_status[user_id] = False

    return Response(generate(), mimetype='text/event-stream')


def process_user_queue(user_id):
    continuous_log_index = 0  # Maintain continuous indexing across files

    try:
        while user_file_queues[user_id]:
            # Process the next file in the queue
            file_info = user_file_queues[user_id].popleft()
            current_file_path = file_info['path']
            original_filename = file_info['original_name']

            print(f"[Backend] Processing file: {current_file_path}")
            yield f"data: {json.dumps({'status': 'file_started', 'filename': original_filename})}\n\n"

            is_pcap = current_file_path.lower().endswith('.pcap')
            temp_csv_path = None

            try:
                if is_pcap:
                    csv_filename = f"{uuid.uuid4()}.csv"
                    temp_csv_path = os.path.join('Uploads', csv_filename)
                    backend_dir = os.path.dirname(os.path.dirname(__file__))
                    cicflowmeter_dir = os.path.join(backend_dir, 'cicflowmeter')

                    if platform.system() == 'Windows':
                        venv_activate = os.path.join(cicflowmeter_dir, '.venv', 'Scripts', 'activate.bat')
                    else:
                        venv_activate = os.path.join(cicflowmeter_dir, '.venv', 'bin', 'activate')

                    print(f"[Backend] Running CICFlowMeter for {current_file_path}")
                    if platform.system() == 'Windows':
                        cmd = f'"{venv_activate}" && cicflowmeter -f "{current_file_path}" -c "{temp_csv_path}"'
                    else:
                        cmd = f'. "{venv_activate}" && cicflowmeter -f "{current_file_path}" -c "{temp_csv_path}"'

                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
                    print(f"[Backend] CICFlowMeter STDOUT: {result.stdout}")
                    print(f"[Backend] CICFlowMeter STDERR: {result.stderr}")
                    print(f"[Backend] CICFlowMeter Return code: {result.returncode}")

                    if result.returncode != 0:
                        yield f"data: {json.dumps({'error': f'CICFlowMeter failed for {original_filename}'})}\n\n"
                        continue

                    df = pd.read_csv(temp_csv_path)
                    df['Flow ID'] = 0
                    df = df.rename(columns=column_mapping)
                    new_df = pd.DataFrame({col: df.get(col, 0) for col in desired_columns})
                    feature_list = [parse_csv_row(row) for _, row in new_df.iterrows()]
                    raw_logs = [row.to_json() for _, row in new_df.iterrows()]
                else:
                    print(f"[Backend] Reading CSV file: {current_file_path}")
                    with open(current_file_path, 'r', encoding='utf-8') as f:
                        lines = f.read().strip().split('\n')

                    feature_list = []
                    raw_logs = []
                    for line in lines:
                        if line.strip():
                            features = parse_log(line)
                            feature_list.append(features)
                            raw_logs.append(line)

                if not feature_list:
                    print(f"[Backend] Error: Invalid file data in {current_file_path}")
                    yield f"data: {json.dumps({'error': f'Invalid file data in {original_filename}'})}\n\n"
                    continue

                # Process each log entry with continuous indexing
                for i, (features, log_data) in enumerate(zip(feature_list, raw_logs)):
                    X = np.array(features).reshape(1, -1)
                    feature_columns = scaler.feature_names_in_
                    X_df = pd.DataFrame(X, columns=feature_columns)
                    X_scaled = scaler.transform(X_df)
                    X_pca = pca.transform(X_scaled)
                    X_tensor = torch.tensor(X_pca, dtype=torch.float32).unsqueeze(1).to(device)

                    with torch.no_grad():
                        outputs = model(X_tensor)
                        pred_idx = torch.argmax(outputs, dim=1).cpu().item()
                        stage_label = label_encoder.inverse_transform([pred_idx])[0]

                    yield f"data: {json.dumps({'log_index': continuous_log_index, 'stage_label': stage_label, 'filename': original_filename, 'file_log_index': i, 'queue_remaining': len(user_file_queues[user_id])})}\n\n"

                    continuous_log_index += 1
                    time.sleep(0.5)  # 0.5s delay for real-time simulation

                # Send file completion signal
                yield f"data: {json.dumps({'status': 'file_completed', 'filename': original_filename, 'total_logs': len(feature_list)})}\n\n"

            except Exception as e:
                print(f"[Backend] Error processing file {current_file_path}: {str(e)}")
                yield f"data: {json.dumps({'error': f'Error processing {original_filename}: {str(e)}'})}\n\n"
            finally:
                # Clean up files
                for path in [current_file_path, temp_csv_path]:
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                            print(f"[Backend] Cleaned up file: {path}")
                        except OSError as e:
                            print(f"[Backend] Failed to remove file {path}: {str(e)}")

        # All files processed
        yield f"data: {json.dumps({'status': 'all_completed', 'total_logs_processed': continuous_log_index})}\n\n"

    except Exception as e:
        print(f"[Backend] Error in process_user_queue: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        # Clear user processing status
        user_processing_status[user_id] = False
        if user_id in user_file_queues:
            del user_file_queues[user_id]
            print(f"[Backend] Cleared queue for user: {user_id}")


@upload.route('/api/queue-status', methods=['GET'])
@jwt_required()
def get_queue_status():
    current_user = get_current_user()

    if isinstance(current_user, dict):
        user_id = current_user.get('user_id')
    else:
        user_id = current_user

    if not user_id:
        return jsonify({'error': 'Invalid user identifier'}), 400

    # Get the user's queue, default to an empty deque if not exists
    user_queue = user_file_queues.get(user_id, deque())
    queue_length = len(user_queue)
    is_processing = user_processing_status.get(user_id, False)

    # Build list of queued file names
    queue_files = [file_info['original_name'] for file_info in user_queue]

    print(
        f"[Backend] Queue status for user {user_id}: length={queue_length}, is_processing={is_processing}, queue_files={queue_files}")

    return jsonify({
        'queue_length': queue_length,
        'is_processing': is_processing,
        'queue_files': queue_files
    })