from flask import Blueprint, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity, get_jwt
from models import db, User
import bcrypt
from datetime import timedelta

auth = Blueprint('auth', __name__)

# ====== JWT SETUP ======
jwt = JWTManager()

@auth.route('/api/login', methods=['POST', 'OPTIONS'])
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
        identity=str(user.id),
        additional_claims={
            'user_id': user.id,
            'role': user.role
        },
        expires_delta=timedelta(days=1)
    )

    print(f"[Login] Login successful for {username}")
    return jsonify({
        'message': 'Login successful',
        'access_token': access_token,
        'user': {'id': user.id, 'username': user.username, 'role': user.role}
    }), 200

@auth.route('/api/register', methods=['POST', 'OPTIONS'])
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

def get_current_user():
    claims = get_jwt()
    return {
        'user_id': claims.get('user_id'),
        'role': claims.get('role')
    }

@auth.route('/api/profile', methods=['GET', 'OPTIONS'])
@jwt_required()
def get_profile():
    print(f"[Profile] Handling {request.method} request for /profile")
    if request.method == 'OPTIONS':
        print("[Profile] Returning 200 for OPTIONS /profile")
        return '', 200

    try:
        current_user = get_current_user()
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

@auth.route('/api/profile', methods=['PUT', 'OPTIONS'])
@jwt_required()
def update_profile():
    print(f"[Profile] Handling {request.method} request for /profile")
    if request.method == 'OPTIONS':
        print("[Profile] Returning 200 for OPTIONS /profile")
        return '', 200

    try:
        current_user = get_current_user()
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

@auth.route('/api/change-password', methods=['POST', 'OPTIONS'])
@jwt_required()
def change_password():
    print(f"[Change-Password] Handling {request.method} request for /change-password")
    if request.method == 'OPTIONS':
        print("[Change-Password] Returning 200 for OPTIONS /change-password")
        return '', 200

    try:
        current_user = get_current_user()
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