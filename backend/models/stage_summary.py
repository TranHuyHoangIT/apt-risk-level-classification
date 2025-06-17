from . import db

class StageSummary(db.Model):
    __tablename__ = 'stage_summary'
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    upload_id = db.Column(db.BigInteger, db.ForeignKey('uploads.id', ondelete='CASCADE'), nullable=True, index=True)
    stage_label = db.Column(db.String(100), nullable=True)
    count = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    __table_args__ = (db.UniqueConstraint('upload_id', 'stage_label', name='uix_upload_stage'),)

    def __repr__(self):
        return f'<StageSummary {self.stage_label}>'