from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class CompletionStatus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    status = db.Column(db.String(255), nullable=False)

class Class(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)

class DetectionEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    detection_time = db.Column(db.DateTime, default=datetime.utcnow)
    completion_status_id = db.Column(db.Integer, db.ForeignKey('completion_status.id'))

class DetectionEventClass(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    detection_event_id = db.Column(db.Integer, db.ForeignKey('detection_event.id'))
    class_id = db.Column(db.Integer, db.ForeignKey('class.id'))
