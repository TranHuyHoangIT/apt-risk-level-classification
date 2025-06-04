from . import db

class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    upload_id = db.Column(db.BigInteger, db.ForeignKey('uploads.id', ondelete='CASCADE'), nullable=True, index=True)
    log_data = db.Column(db.Text, nullable=True)
    predicted_label = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

    def __repr__(self):
        return f'<Prediction {self.id}>'