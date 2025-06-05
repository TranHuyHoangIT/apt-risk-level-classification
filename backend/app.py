import os
from datetime import timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from models import db
from flask_socketio import SocketIO

app = Flask(__name__)

# Cấu hình CORS
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Xử lý yêu cầu OPTIONS thủ công để đảm bảo CORS
@app.before_request
def handle_options():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'CORS preflight successful'})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response, 200

#Ubuntu
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://hoangtran:04112003thh#@localhost/apt'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.urandom(24).hex()
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=1)

db.init_app(app)
jwt = JWTManager(app)
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:3000"], async_mode='eventlet')  # Sử dụng eventlet cho WebSocket

# Kiểm tra import các blueprint từ thư mục routes
try:
    from routes import auth, upload, user, model_bp, realtime
    from routes.realtime import init_socketio  # Import hàm init_socketio
    print("[Startup] Imported blueprints successfully")
except ImportError as e:
    print(f"[Startup Error] Failed to import blueprints: {e}")
    raise

# Đăng ký các blueprint
print("[Startup] Registering blueprints")
app.register_blueprint(auth, url_prefix='')
app.register_blueprint(upload, url_prefix='')
app.register_blueprint(user, url_prefix='')
app.register_blueprint(model_bp, url_prefix='')
app.register_blueprint(realtime, url_prefix='')
init_socketio(socketio)  # Khởi tạo sự kiện SocketIO
print("[Startup] Blueprints registered")

if __name__ == "__main__":
    with app.app_context():
        print("[Startup] Creating database tables")
        try:
            db.create_all()
            print("[Startup] Database tables created")
        except Exception as e:
            print(f"[Startup Error] Database error: {e}")
    print("[Startup] Starting Flask server")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)