import os
import bcrypt
from datetime import timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from models import db, User
from dotenv import load_dotenv

# Load environment variables from .env file
env = os.getenv('FLASK_ENV', 'development')

# Choose dotenv file based on environment
if env == 'production':
    dotenv_path = '.env.prod'
else:
    dotenv_path = '.env.dev'

load_dotenv(dotenv_path=dotenv_path)
print(f"[ENV] Loaded config from: {dotenv_path}")
print("DATABASE_URL:", os.getenv('DATABASE_URL'))
print("FRONTEND_URL", os.getenv('FRONTEND_URL'))

app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": [os.getenv('FRONTEND_URL', 'http://localhost:3000')],
        "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Configure database and JWT
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=45)

db.init_app(app)
jwt = JWTManager(app)

# Import and register blueprints
try:
    from routes import auth, upload, user, model_bp
    print("[Startup] Imported blueprints successfully")
except ImportError as e:
    print(f"[Startup Error] Failed to import blueprints: {e}")
    raise

print("[Startup] Registering blueprints")
app.register_blueprint(auth, url_prefix='')
app.register_blueprint(upload, url_prefix='')
app.register_blueprint(user, url_prefix='')
app.register_blueprint(model_bp, url_prefix='')
print("[Startup] Blueprints registered")

# Initialize database and create sample users
def init_db():
    with app.app_context():
        print("[Startup] Creating database tables")
        try:
            db.create_all()
            print("[Startup] Database tables created")
            
            if User.query.count() == 0:
                print("[Startup] Inserting sample users")

                admin = User(
                    username='admin',
                    email='admin@gmail.com',
                    password_hash=bcrypt.hashpw('admin@123'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
                    role='admin'
                )
                user1 = User(
                    username='user1',
                    email='user1@gmail.com',
                    password_hash=bcrypt.hashpw('12345678'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
                    role='user'
                )

                db.session.add_all([admin, user1])
                db.session.commit()
                print("[Startup] Sample users inserted")
        except Exception as e:
            print(f"[Startup Error] Database error: {e}")
            raise

init_db()

if __name__ == '__main__':
    port = int(os.getenv('BACKEND_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    print("[Startup] Starting Flask server")
    app.run(debug=debug, host='0.0.0.0', port=port)