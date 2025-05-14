from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Upload(db.Model):
    __tablename__ = 'uploads'
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    filename = db.Column(db.String(255))
    file_path = db.Column(db.String(500))
    upload_time = db.Column(db.DateTime, server_default=db.func.now())
    predictions = db.relationship('Prediction', backref='upload', cascade="all, delete")
    risk_summaries = db.relationship('RiskSummary', backref='upload', cascade="all, delete")

class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    upload_id = db.Column(db.BigInteger, db.ForeignKey('uploads.id', ondelete='CASCADE'))
    log_data = db.Column(db.Text)
    predicted_label = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, server_default=db.func.now())

class RiskSummary(db.Model):
    __tablename__ = 'risk_summary'
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    upload_id = db.Column(db.BigInteger, db.ForeignKey('uploads.id', ondelete='CASCADE'))
    risk_level = db.Column(db.String(100))
    count = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
