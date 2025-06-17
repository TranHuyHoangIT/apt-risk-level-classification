from . import db

class Upload(db.Model):
    __tablename__ = 'uploads'
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    filename = db.Column(db.String(255), nullable=True)
    file_path = db.Column(db.String(500), nullable=True)
    user_id = db.Column(db.BigInteger, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    upload_time = db.Column(db.DateTime, server_default=db.func.now())
    predictions = db.relationship('Prediction', backref='upload', cascade="all, delete", lazy='dynamic')
    stage_summaries = db.relationship('StageSummary', backref='upload', cascade="all, delete", lazy='dynamic')

    def __repr__(self):
        return f'<Upload {self.filename}>'