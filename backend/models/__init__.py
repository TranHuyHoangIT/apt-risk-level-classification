from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

from .user import User
from .upload import Upload
from .prediction import Prediction
from .risk_summary import RiskSummary

__all__ = ['db', 'User', 'Upload', 'Prediction', 'RiskSummary']