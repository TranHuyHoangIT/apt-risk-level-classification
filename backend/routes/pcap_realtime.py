from flask import Blueprint, jsonify
from flask_socketio import emit
from scapy.all import sniff, IP
from routes.model import model, scaler, label_encoder, parse_log, device
import numpy as np
import torch
import subprocess
import threading
import logging
import os
import time
import sys
import json

pcap_realtime = Blueprint('pcap_realtime', __name__)

# Thiết lập logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global variables để quản lý trạng thái
sniffer_process = None
tcpreplay_thread = None
is_running = False
socketio_instance = None

def run_tcpreplay():
    """Hàm chạy tcpreplay trong thread riêng"""
    global is_running
    pcap_file = "enp0s3-tcpdump-friday.pcap"
    
    if not os.path.exists(pcap_file):
        error_msg = f"File PCAP không tồn tại: {pcap_file}"
        logger.error(error_msg)
        if socketio_instance:
            socketio_instance.emit('pcap_error', {'message': error_msg})
        return

    logger.info("Bắt đầu phát lại file PCAP với tcpreplay")
    
    try:
        # Thêm delay nhỏ để đảm bảo sniffer đã sẵn sàng
        time.sleep(3)
        
        process = subprocess.Popen(
            ["sudo", "tcpreplay", "-i", "lo", "--pps", "10", pcap_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            logger.info("Tcpreplay hoàn tất thành công")
            if socketio_instance:
                socketio_instance.emit('pcap_status', {'message': 'PCAP replay completed'})
        else:
            error_msg = f"Lỗi tcpreplay: {stderr}"
            logger.error(error_msg)
            if socketio_instance:
                socketio_instance.emit('pcap_error', {'message': error_msg})
                
    except Exception as e:
        error_msg = f"Exception khi chạy tcpreplay: {str(e)}"
        logger.error(error_msg)
        if socketio_instance:
            socketio_instance.emit('pcap_error', {'message': error_msg})
    finally:
        is_running = False

def create_sniffer_script():
    """Tạo script Python để chạy sniffer với sudo"""
    script_content = f'''#!/usr/bin/env python3
import sys
import os
import json
import time
import numpy as np
import torch
from scapy.all import sniff, IP

# Add the parent directory to sys.path để import được modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from routes.model import model, scaler, label_encoder, parse_log, device
except ImportError as e:
    print(f"Import error: {{e}}", file=sys.stderr)
    sys.exit(1)

def process_packet(packet):
    """Xử lý từng gói tin và gửi kết quả qua stdout"""
    if IP in packet:
        try:
            # Tạo log string từ packet
            log_str = f"{{packet.time}},{{packet[IP].src}},{{packet[IP].dst}},{{packet[IP].proto}},0,0,0,{{packet[IP].len}}"
            
            # Parse features
            features = parse_log(log_str)
            X = np.array([features])
            X_scaled = scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(device)

            # Dự đoán
            with torch.no_grad():
                outputs = model(X_tensor)
                pred = torch.argmax(outputs, dim=1).cpu().numpy()
                risk_label = label_encoder.inverse_transform(pred)[0]

            # Tạo thông tin gói tin
            packet_info = {{
                'src_ip': packet[IP].src,
                'dst_ip': packet[IP].dst,
                'protocol': packet[IP].proto,
                'length': packet[IP].len,
                'timestamp': float(packet.time),
                'risk_level': risk_label
            }}
            
            # Gửi kết quả qua stdout
            print(json.dumps(packet_info), flush=True)
            
        except Exception as e:
            error_info = {{'error': f'Lỗi khi xử lý gói tin: {{str(e)}}'}}
            print(json.dumps(error_info), file=sys.stderr, flush=True)

def main():
    try:
        print(json.dumps({{'status': 'Sniffer started'}}), flush=True)
        
        # Bắt đầu sniff
        sniff(
            iface="lo", 
            prn=process_packet, 
            store=0,
            timeout=1
        )
        
    except KeyboardInterrupt:
        print(json.dumps({{'status': 'Sniffer stopped by interrupt'}}), flush=True)
    except Exception as e:
        error_info = {{'error': f'Lỗi khi chạy sniffer: {{str(e)}}'}}
        print(json.dumps(error_info), file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
'''
    
    # Tạo file script tạm thời
    script_path = "/tmp/packet_sniffer.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    return script_path

def start_sniffer():
    """Khởi động sniffer process với sudo"""
    global sniffer_process, is_running
    
    try:
        # Tạo script sniffer
        script_path = create_sniffer_script()
        logger.info(f"Created sniffer script at: {script_path}")
        
        # Chạy script với sudo
        sniffer_process = subprocess.Popen(
            ["sudo", "python3", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        logger.info("Sniffer process started with sudo")
        
        # Đọc output từ sniffer process
        while is_running and sniffer_process.poll() is None:
            try:
                # Đọc stdout
                line = sniffer_process.stdout.readline()
                if line:
                    try:
                        packet_data = json.loads(line.strip())
                        if 'error' not in packet_data and socketio_instance:
                            socketio_instance.emit('packet_data', packet_data)
                            logger.debug(f"Emitted packet: {packet_data.get('src_ip')} -> {packet_data.get('dst_ip')}, Risk: {packet_data.get('risk_level')}")
                        elif 'status' in packet_data:
                            logger.info(f"Sniffer status: {packet_data['status']}")
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse sniffer output: {line.strip()}")
                
                # Đọc stderr
                if sniffer_process.stderr:
                    stderr_line = sniffer_process.stderr.readline()
                    if stderr_line:
                        try:
                            error_data = json.loads(stderr_line.strip())
                            if 'error' in error_data:
                                logger.error(f"Sniffer error: {error_data['error']}")
                                if socketio_instance:
                                    socketio_instance.emit('pcap_error', {'message': error_data['error']})
                        except json.JSONDecodeError:
                            logger.error(f"Sniffer stderr: {stderr_line.strip()}")
                            
            except Exception as e:
                logger.error(f"Error reading sniffer output: {str(e)}")
                break
        
        # Cleanup
        if sniffer_process and sniffer_process.poll() is None:
            sniffer_process.terminate()
            sniffer_process.wait(timeout=5)
        
        # Remove temporary script
        try:
            os.unlink(script_path)
        except:
            pass
            
        logger.info("Sniffer process ended")
        
    except Exception as e:
        error_msg = f"Lỗi khi khởi động sniffer: {str(e)}"
        logger.error(error_msg)
        if socketio_instance:
            socketio_instance.emit('pcap_error', {'message': error_msg})

@pcap_realtime.route('/start-pcap', methods=['POST'])
def start_pcap():
    """API endpoint để bắt đầu xử lý PCAP"""
    global sniffer_process, tcpreplay_thread, is_running
    
    if is_running:
        return jsonify({'message': 'PCAP processing đã đang chạy'}), 400
    
    try:
        is_running = True
        
        # Khởi động sniffer thread
        sniffer_thread = threading.Thread(target=start_sniffer, daemon=True)
        sniffer_thread.start()
        logger.info("Sniffer thread started")
        
        # Khởi động tcpreplay thread
        tcpreplay_thread = threading.Thread(target=run_tcpreplay, daemon=True)
        tcpreplay_thread.start()
        logger.info("Tcpreplay thread started")
        
        return jsonify({'message': 'Bắt đầu xử lý PCAP theo thời gian thực'})
        
    except Exception as e:
        is_running = False
        error_msg = f"Lỗi khi khởi động PCAP processing: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

@pcap_realtime.route('/stop-pcap', methods=['POST'])
def stop_pcap():
    """API endpoint để dừng xử lý PCAP"""
    global is_running, sniffer_process
    
    is_running = False
    logger.info("Đã yêu cầu dừng PCAP processing")
    
    # Terminate sniffer process nếu đang chạy
    if sniffer_process and sniffer_process.poll() is None:
        try:
            sniffer_process.terminate()
            sniffer_process.wait(timeout=5)
            logger.info("Sniffer process terminated")
        except subprocess.TimeoutExpired:
            sniffer_process.kill()
            logger.info("Sniffer process killed")
        except Exception as e:
            logger.error(f"Error stopping sniffer process: {str(e)}")
    
    if socketio_instance:
        socketio_instance.emit('pcap_status', {'message': 'PCAP processing stopped'})
    
    return jsonify({'message': 'Đã dừng xử lý PCAP'})

@pcap_realtime.route('/pcap-status', methods=['GET'])
def get_pcap_status():
    """API endpoint để kiểm tra trạng thái PCAP"""
    sniffer_active = sniffer_process and sniffer_process.poll() is None if sniffer_process else False
    tcpreplay_active = tcpreplay_thread and tcpreplay_thread.is_alive() if tcpreplay_thread else False
    
    return jsonify({
        'is_running': is_running,
        'sniffer_active': sniffer_active,
        'tcpreplay_active': tcpreplay_active
    })

def init_socketio(socketio_inst):
    """Khởi tạo SocketIO handlers"""
    global socketio_instance
    socketio_instance = socketio_inst
    
    @socketio_instance.on('connect')
    def handle_connect():
        logger.info("Client đã kết nối qua WebSocket")
        socketio_instance.emit('connection_status', {
            'message': 'Connected to PCAP monitoring server',
            'timestamp': time.time()
        })
    
    @socketio_instance.on('disconnect')
    def handle_disconnect():
        logger.info("Client đã ngắt kết nối WebSocket")
    
    @socketio_instance.on('ping')
    def handle_ping():
        socketio_instance.emit('pong', {'timestamp': time.time()})