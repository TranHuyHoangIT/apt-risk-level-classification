from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

from .user import User
from .upload import Upload
from .prediction import Prediction
from .stage_summary import StageSummary

__all__ = ['db', 'User', 'Upload', 'Prediction', 'StageSummary']