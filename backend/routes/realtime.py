from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask_socketio import emit, join_room, leave_room
import os
import subprocess
import uuid
import platform
import pandas as pd
import numpy as np
import torch
import threading
import time
import logging
from .model import model, scaler, label_encoder, parse_csv_row, desired_columns, column_mapping, device

# Thiết lập logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

realtime = Blueprint('realtime', __name__)

# Global variables for realtime monitoring
monitoring_sessions = {}

class RealtimeMonitor:
    def __init__(self, pcap_file, session_id, socketio):
        self.pcap_file = pcap_file
        self.session_id = session_id
        self.socketio = socketio
        self.is_running = False
        self.thread = None
        self.replay_process = None
        self.temp_dir = f"/home/hoangtran/apt-risk-level-classification/backend/temp/{session_id}"
        os.makedirs(self.temp_dir, exist_ok=True)
        os.chmod(self.temp_dir, 0o777)  # Đảm bảo quyền 777 cho thư mục
        
    def start_monitoring(self):
        """Start realtime monitoring"""
        if self.is_running:
            logger.warning(f"[Session {self.session_id}] Monitoring already running")
            return False
            
        self.is_running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"[Session {self.session_id}] Started monitoring thread")
        return True
    
    def stop_monitoring(self):
        """Stop realtime monitoring"""
        self.is_running = False
        if self.replay_process:
            try:
                logger.info(f"[Session {self.session_id}] Terminating tcpreplay process")
                self.replay_process.terminate()
                self.replay_process.wait(timeout=5)
            except:
                if self.replay_process.poll() is None:
                    logger.warning(f"[Session {self.session_id}] Force killing tcpreplay process")
                    self.replay_process.kill()
        if os.path.exists(self.pcap_file):
            logger.debug(f"[Session {self.session_id}] Removing PCAP file: {self.pcap_file}")
            os.remove(self.pcap_file)
        if os.path.exists(self.temp_dir):
            logger.debug(f"[Session {self.session_id}] Removing temp directory: {self.temp_dir}")
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)

    def _monitor_loop(self):
        """Main monitoring loop"""
        try:
            logger.debug(f"[Session {self.session_id}] Using temp directory: {self.temp_dir}")
            
            # Đọc toàn bộ PCAP và chia thành chunks
            total_packets = self._get_total_packets()
            if total_packets == 0:
                logger.warning(f"[Session {self.session_id}] No packets found in PCAP file")
                return
                
            chunk_size = 10
            processed_packets = 0
            
            while self.is_running and processed_packets < total_packets:
                temp_pcap = os.path.join(self.temp_dir, f"chunk_{processed_packets}.pcap")
                
                if self._extract_pcap_chunk(temp_pcap, processed_packets, chunk_size):
                    risk_data = self._process_pcap_chunk(temp_pcap)
                    
                    if risk_data:
                        logger.info(f"[Session {self.session_id}] Emitting risk data: {risk_data}")
                        self.socketio.emit('realtime_risk_update', {
                            'session_id': self.session_id,
                            'timestamp': time.time(),
                            'risk_data': risk_data,
                            'processed_packets': processed_packets + chunk_size
                        }, room=self.session_id)
                        
                    processed_packets += chunk_size
                    time.sleep(2)  # Delay giữa các chunk
                else:
                    logger.warning(f"[Session {self.session_id}] Failed to extract chunk at packet {processed_packets}")
                    break
                    
        except Exception as e:
            logger.error(f"[Session {self.session_id}] Error in monitor loop: {str(e)}")
            self.socketio.emit('realtime_error', {
                'session_id': self.session_id,
                'error': str(e)
            }, room=self.session_id)
        finally:
            self.stop_monitoring()

    def _get_total_packets(self):
        """Get total number of packets in PCAP file"""
        try:
            cmd = ["tshark", "-r", self.pcap_file, "-T", "fields", "-e", "frame.number"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                return len([line for line in lines if line.strip()])
            return 0
        except Exception as e:
            logger.error(f"Error getting packet count: {e}")
            return 0
    
    def _start_tcpreplay(self):
        """Start tcpreplay to simulate network traffic"""
        try:
            cmd = ["sudo", "tcpreplay", "-i", "lo", "-M", "0.5", self.pcap_file]
            logger.debug(f"[Session {self.session_id}] Starting tcpreplay with command: {cmd}")
            self.replay_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"[Session {self.session_id}] Started tcpreplay with PID: {self.replay_process.pid}")
            time.sleep(1)
            if self.replay_process.poll() is not None:
                stdout, stderr = self.replay_process.communicate()
                logger.error(f"[Session {self.session_id}] tcpreplay failed immediately. stdout: {stdout}, stderr: {stderr}")
                raise RuntimeError("tcpreplay failed to start")
        except Exception as e:
            logger.error(f"[Session {self.session_id}] Failed to start tcpreplay: {str(e)}")
            raise

    def _extract_pcap_chunk(self, output_file, start_packet, count):
        """Extract a chunk of packets from PCAP file"""
        try:
            if not os.path.exists(self.pcap_file):
                logger.error(f"[Session {self.session_id}] Input PCAP file not found: {self.pcap_file}")
                return False
            
            # Sử dụng tshark thay vì editcap để extract packets
            cmd = [
                "tshark", 
                "-r", self.pcap_file,
                "-w", output_file,
                "-c", str(count)
            ]
            
            if start_packet > 0:
                cmd.extend(["-Y", f"frame.number >= {start_packet + 1}"])
            
            logger.debug(f"[Session {self.session_id}] Running tshark command: {cmd}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"[Session {self.session_id}] tshark failed: {result.stderr}")
                return False
                
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                logger.warning(f"[Session {self.session_id}] No packets extracted")
                return False
                
            logger.info(f"[Session {self.session_id}] Successfully extracted chunk: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"[Session {self.session_id}] Error extracting pcap chunk: {str(e)}")
            return False
    
    def _process_pcap_chunk(self, pcap_file):
        """Process a pcap chunk and return risk analysis"""
        try:
            csv_filename = f"{uuid.uuid4()}.csv"
            csv_file = os.path.join('Uploads', csv_filename)
            refactored_csv_path = os.path.join('Uploads', f"refactored_{csv_filename}")
            os.makedirs('Uploads', exist_ok=True)

            cicflowmeter_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cicflowmeter')
            venv_activate = os.path.join(cicflowmeter_dir, '.venv', 'bin', 'activate') if platform.system() != 'Windows' else os.path.join(cicflowmeter_dir, '.venv', 'Scripts')
            # cicflowmeter_path = os.path.join(venv_bin, 'cicflowmeter') if platform.system() != 'Windows' else os.path.join(venv_bin, 'cicflowmeter.exe')

            # if not os.path.exists(cicflowmeter_path):
            #     logger.error(f"[Session {self.session_id}] CICFlowMeter not found at {cicflowmeter_path}")
            #     return None

            # cmd = [cicflowmeter_path, "-f", pcap_file, "-c", csv_file]
            cmd = f'. "{venv_activate}" && cicflowmeter -f "{pcap_file}" -c "{csv_file}"'
            logger.debug(f"[Session {self.session_id}] Running CICFlowMeter: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            logger.debug(f"[Session {self.session_id}] CICFlowMeter stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"[Session {self.session_id}] CICFlowMeter stderr: {result.stderr}")
            if result.returncode != 0:
                logger.error(f"[Session {self.session_id}] CICFlowMeter failed with return code {result.returncode}")
                return None

            if not os.path.exists(csv_file):
                logger.warning(f"[Session {self.session_id}] CSV file not created: {csv_file}")
                return None

            df = pd.read_csv(csv_file)
            logger.debug(f"[Session {self.session_id}] CSV data shape: {df.shape}")
            if df.empty:
                logger.warning(f"[Session {self.session_id}] CSV file is empty")
                return None

            df['Flow ID'] = 0
            df = df.rename(columns=column_mapping)
            new_df = pd.DataFrame({col: df.get(col, 0) for col in desired_columns})

            new_df.to_csv(refactored_csv_path, index=False, header=False)
            logger.debug(f"[Session {self.session_id}] Saved refactored CSV: {refactored_csv_path}")

            feature_list = [parse_csv_row(row) for _, row in new_df.iterrows()]
            if not feature_list:
                logger.warning(f"[Session {self.session_id}] No features extracted from CSV")
                return None

            X = np.vstack(feature_list)
            X_scaled = scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(device)
            
            with torch.no_grad():
                outputs = model(X_tensor)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                risk_labels = label_encoder.inverse_transform(preds)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                max_probs = np.max(probs, axis=1)
            
            risk_count = {}
            for label in risk_labels:
                risk_count[label] = risk_count.get(label, 0) + 1
            
            avg_confidence = float(np.mean(max_probs))
            
            for temp_file in [pcap_file, csv_file, refactored_csv_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            logger.debug(f"[Session {self.session_id}] Risk distribution: {risk_count}, Avg confidence: {avg_confidence}")
            return {
                'risk_distribution': risk_count,
                'total_flows': len(risk_labels),
                'avg_confidence': avg_confidence,
                'high_risk_flows': sum(1 for label in risk_labels if label in ['Cao', 'Rất cao'])
            }
            
        except Exception as e:
            logger.error(f"[Session {self.session_id}] Error processing pcap chunk: {str(e)}")
            return None

@realtime.route('/start-realtime', methods=['POST'])
@jwt_required()
def start_realtime_monitoring():
    """Start realtime monitoring of pcap file"""
    current_user = get_jwt_identity()
    
    if 'file' not in request.files:
        logger.error("[API] Missing pcap file in request")
        return jsonify({'error': 'Missing pcap file'}), 400
    
    file = request.files['file']
    if not file.filename.lower().endswith('.pcap'):
        logger.error(f"[API] Invalid file format: {file.filename}")
        return jsonify({'error': 'Invalid PCAP file'}), 400
    
    session_id = str(uuid.uuid4())
    pcap_filename = f"realtime_{session_id}.pcap"
    pcap_path = os.path.join('Uploads', pcap_filename)
    os.makedirs('Uploads', exist_ok=True)
    file.save(pcap_path)
    logger.info(f"[Session {session_id}] Saved PCAP file to: {pcap_path}")
    
    from app import socketio
    monitor = RealtimeMonitor(pcap_path, session_id, socketio)
    monitoring_sessions[session_id] = monitor
    
    if monitor.start_monitoring():
        logger.info(f"[Session {session_id}] Realtime monitoring started")
        return jsonify({
            'session_id': session_id,
            'status': 'started',
            'message': 'Realtime monitoring started'
        })
    else:
        logger.error(f"[Session {session_id}] Failed to start monitoring")
        return jsonify({'error': 'Failed to start monitoring'}), 500

@realtime.route('/stop-realtime/<session_id>', methods=['POST'])
@jwt_required()
def stop_realtime_monitoring(session_id):
    """Stop realtime monitoring"""
    if session_id in monitoring_sessions:
        logger.info(f"[Session {session_id}] Stopping monitoring")
        monitoring_sessions[session_id].stop_monitoring()
        return jsonify({'status': 'stopped', 'message': 'Monitoring stopped'})
    else:
        logger.error(f"[Session {session_id}] Session not found")
        return jsonify({'error': 'Session not found'}), 404

@realtime.route('/realtime-status', methods=['GET'])
@jwt_required()
def get_realtime_status():
    """Get status of all active monitoring sessions"""
    active_sessions = [
        {
            'session_id': session_id,
            'is_running': monitor.is_running,
            'pcap_file': os.path.basename(monitor.pcap_file)
        }
        for session_id, monitor in monitoring_sessions.items()
    ]
    logger.debug(f"[API] Active sessions: {active_sessions}")
    return jsonify({'active_sessions': active_sessions})

def init_socketio(socketio_instance):
    """Initialize SocketIO events"""
    @socketio_instance.on('connect')
    def on_connect():
        logger.info("Client connected to SocketIO")

    @socketio_instance.on('join')
    def on_join(session_id):
        logger.debug(f"Client joined session: {session_id}")
        join_room(session_id)
        emit('join_success', {'message': f'Joined session {session_id}'}, room=session_id)
    
    @socketio_instance.on('leave')
    def on_leave(session_id):
        logger.debug(f"Client left session: {session_id}")
        leave_room(session_id)
        emit('leave_success', {'message': f'Left session {session_id}'}, room=session_id)