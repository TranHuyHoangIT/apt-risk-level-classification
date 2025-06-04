from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import db, User
import bcrypt

user = Blueprint('user', __name__)

@user.route('/users', methods=['GET', 'OPTIONS'])
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

@user.route('/users/<int:user_id>', methods=['GET', 'OPTIONS'])
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

        user_entry = User.query.get_or_404(user_id)
        print(f"[User] Fetched user {user_entry.username}")
        return jsonify({
            'id': user_entry.id,
            'username': user_entry.username,
            'email': user_entry.email,
            'role': user_entry.role,
            'created_at': user_entry.created_at.isoformat()
        }), 200
    except Exception as e:
        print(f"[User] Error: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@user.route('/users/<int:user_id>', methods=['PUT', 'OPTIONS'])
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

        user_entry = User.query.get_or_404(user_id)
        data = request.get_json()
        print(f"[User] Update data for user {user_id}: {data}")

        username = data.get('username')
        email = data.get('email')
        new_password = data.get('newPassword')

        if not username or not email:
            print("[User] Missing username or email")
            return jsonify({'error': 'Missing username or email'}), 400

        if username != user_entry.username and User.query.filter_by(username=username).first():
            print(f"[User] Username {username} already exists")
            return jsonify({'error': 'Username already exists'}), 400
        if email != user_entry.email and User.query.filter_by(email=email).first():
            print(f"[User] Email {email} already exists")
            return jsonify({'error': 'Email already exists'}), 400

        user_entry.username = username
        user_entry.email = email
        if new_password:
            user_entry.password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        db.session.commit()
        print(f"[User] Updated user {user_entry.username}")
        return jsonify({
            'id': user_entry.id,
            'username': user_entry.username,
            'email': user_entry.email,
            'role': user_entry.role,
            'created_at': user_entry.created_at.isoformat()
        }), 200
    except Exception as e:
        print(f"[User] Error: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@user.route('/users/<int:user_id>', methods=['DELETE', 'OPTIONS'])
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

        user_entry = User.query.get_or_404(user_id)
        db.session.delete(user_entry)
        db.session.commit()
        print(f"[User] Deleted user {user_id}")
        return jsonify({'message': 'User deleted successfully'}), 200
    except Exception as e:
        print(f"[User] Error: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500