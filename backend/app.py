import os
from datetime import timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from models import db

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

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://TranHuyHoang:04112003thh%23@localhost/apt'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://hoangtran:04112003thh#@localhost/apt'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.urandom(24).hex()
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=1)

db.init_app(app)
jwt = JWTManager(app)

# Kiểm tra import routes
try:
    from routes import routes
    print("[Startup] Imported routes successfully")
except ImportError as e:
    print(f"[Startup Error] Failed to import routes: {e}")
    raise

# Đăng ký blueprint
print("[Startup] Registering blueprint 'routes'")
app.register_blueprint(routes, url_prefix='')
print("[Startup] Blueprint 'routes' registered")

if __name__ == '__main__':
    with app.app_context():
        print("[Startup] Creating database tables")
        try:
            db.create_all()
            print("[Startup] Database tables created")
        except Exception as e:
            print(f"[Startup Error] Database error: {e}")
    print("[Startup] Starting Flask server")
    app.run(debug=True, host='0.0.0.0', port=5000)