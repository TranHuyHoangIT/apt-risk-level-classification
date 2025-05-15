from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.Enum('user', 'admin'), nullable=False, default='user')
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    updated_at = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())
    uploads = db.relationship('Upload', backref='user', cascade="all, delete", lazy='dynamic')

    def __repr__(self):
        return f'<User {self.username}>'

class Upload(db.Model):
    __tablename__ = 'uploads'
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    filename = db.Column(db.String(255), nullable=True)
    file_path = db.Column(db.String(500), nullable=True)
    user_id = db.Column(db.BigInteger, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    upload_time = db.Column(db.DateTime, server_default=db.func.now())
    predictions = db.relationship('Prediction', backref='upload', cascade="all, delete", lazy='dynamic')
    risk_summaries = db.relationship('RiskSummary', backref='upload', cascade="all, delete", lazy='dynamic')

    def __repr__(self):
        return f'<Upload {self.filename}>'

class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    upload_id = db.Column(db.BigInteger, db.ForeignKey('uploads.id', ondelete='CASCADE'), nullable=True, index=True)
    log_data = db.Column(db.Text, nullable=True)
    predicted_label = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

    def __repr__(self):
        return f'<Prediction {self.id}>'

class RiskSummary(db.Model):
    __tablename__ = 'risk_summary'
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    upload_id = db.Column(db.BigInteger, db.ForeignKey('uploads.id', ondelete='CASCADE'), nullable=True, index=True)
    risk_level = db.Column(db.String(100), nullable=True)
    count = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    __table_args__ = (db.UniqueConstraint('upload_id', 'risk_level', name='uix_upload_risk'),)

    def __repr__(self):
        return f'<RiskSummary {self.risk_level}>'