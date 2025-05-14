from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from models import db
from routes import routes

app = Flask(__name__)
CORS(app)

# Cấu hình DB (update theo cấu hình MySQL của bạn)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://TranHuyHoang:04112003thh#@localhost/apt'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
app.register_blueprint(routes)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # tạo bảng nếu chưa có
    app.run(debug=True)
