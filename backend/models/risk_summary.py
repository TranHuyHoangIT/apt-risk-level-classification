from . import db

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