from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pytz

db = SQLAlchemy()

class CompletionStatus(db.Model):
    __tablename__ = 'completion_status'
    id = db.Column(db.Integer, primary_key=True)
    status = db.Column(db.String(255), nullable=False)

class Class(db.Model):
    __tablename__ = 'class'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)

class DetectionEvent(db.Model):
    __tablename__ = 'detection_event'
    id = db.Column(db.Integer, primary_key=True)
    detection_time = db.Column(db.DateTime, default=lambda: datetime.now(pytz.timezone('Asia/Singapore')))
    completion_status_id = db.Column(db.Integer, db.ForeignKey('completion_status.id'), nullable=False)
    classID = db.Column(db.Integer, db.ForeignKey('class.id'), nullable=False)

    completion_status = db.relationship('CompletionStatus', backref=db.backref('detection_events', lazy=True))
    detected_class = db.relationship('Class', backref=db.backref('detection_events', lazy=True))
