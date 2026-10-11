"""REST API blueprint."""
import os
import shutil
from flask import Blueprint, request, jsonify, current_app, send_file
import config

api_bp = Blueprint('api', __name__, url_prefix='/api')

def _db():
    return current_app.db

def _pipeline():
    return current_app.pipeline

@api_bp.route('/recordings')
def list_recordings():
    date = request.args.get('date', '')
    if not date:
        from datetime import datetime
        date = datetime.now().strftime('%Y-%m-%d')
    rows = _db().get_recordings_by_date(date)
    result = []
    for r in rows:
        d = dict(r)
        # Count faces in this recording
        faces = _db().get_face_events_by_recording(r['id'])
        d['face_count'] = len(faces)
        # Make path relative for serving
        d['url'] = '/recordings/' + os.path.relpath(r['file_path'], config.RECORDINGS_DIR)
        result.append(d)
    return jsonify(result)

@api_bp.route('/recordings/<int:rec_id>')
def get_recording(rec_id):
    r = _db().get_recording(rec_id)
    if not r:
        return jsonify({'error': 'not found'}), 404
    d = dict(r)
    d['url'] = '/recordings/' + os.path.relpath(r['file_path'], config.RECORDINGS_DIR)
    faces = _db().get_face_events_by_recording(rec_id)
    d['faces'] = []
    for f in faces:
        fd = dict(f)
        if fd.get('crop_path') and os.path.exists(fd['crop_path']):
            fd['crop_url'] = '/face_crops/' + os.path.relpath(fd['crop_path'], config.FACE_CROPS_DIR)
        fd.pop('embedding', None)
        d['faces'].append(fd)
    return jsonify(d)

@api_bp.route('/faces')
def list_faces():
    groups = _db().get_all_face_groups()
    result = []
    for g in groups:
        d = dict(g)
        d.pop('mean_embedding', None)
        if d.get('representative_crop') and os.path.exists(d['representative_crop']):
            d['crop_url'] = '/face_crops/' + os.path.relpath(d['representative_crop'], config.FACE_CROPS_DIR)
        else:
            d['crop_url'] = None
        result.append(d)
    return jsonify(result)

@api_bp.route('/faces/<int:group_id>')
def get_face_group(group_id):
    g = _db().get_face_group(group_id)
    if not g:
        return jsonify({'error': 'not found'}), 404
    d = dict(g)
    d.pop('mean_embedding', None)
    if d.get('representative_crop') and os.path.exists(d['representative_crop']):
        d['crop_url'] = '/face_crops/' + os.path.relpath(d['representative_crop'], config.FACE_CROPS_DIR)
    events = _db().get_face_events_by_group(group_id)
    d['events'] = []
    for e in events:
        ed = dict(e)
        ed.pop('embedding', None)
        if ed.get('crop_path') and os.path.exists(ed['crop_path']):
            ed['crop_url'] = '/face_crops/' + os.path.relpath(ed['crop_path'], config.FACE_CROPS_DIR)
        d['events'].append(ed)
    return jsonify(d)

@api_bp.route('/faces/<int:group_id>', methods=['PUT'])
def update_face_label(group_id):
    data = request.get_json()
    label = data.get('label', '')
    _db().update_face_group_label(group_id, label)
    return jsonify({'ok': True})

@api_bp.route('/timeline')
def timeline():
    date = request.args.get('date', '')
    if not date:
        from datetime import datetime
        date = datetime.now().strftime('%Y-%m-%d')
    data = _db().get_timeline(date)
    return jsonify(data)

@api_bp.route('/status')
def status():
    stats = _db().get_stats()
    disk = shutil.disk_usage(config.DATA_DIR)
    pipeline = _pipeline()
    return jsonify({
        'pipeline': pipeline.status if pipeline else {},
        'db': stats,
        'disk': {
            'total_gb': round(disk.total / (1024**3), 1),
            'used_gb': round(disk.used / (1024**3), 1),
            'free_gb': round(disk.free / (1024**3), 1),
            'used_pct': round(disk.used / disk.total * 100, 1),
        }
    })
