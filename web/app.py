"""Flask application factory."""
import os
import time
from flask import Flask, send_file, render_template, Response, request, abort
import config

def create_app(db=None, pipeline=None):
    app = Flask(__name__,
                static_folder='static',
                template_folder='templates')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

    # Store refs for API access
    app.db = db
    app.pipeline = pipeline

    from web.api import api_bp
    app.register_blueprint(api_bp)

    def _send_file_partial(path, mimetype=None):
        file_size = os.path.getsize(path)
        range_header = request.headers.get('Range')
        if not range_header:
            return send_file(path, mimetype=mimetype, conditional=True)

        units, _, range_spec = range_header.partition('=')
        if units != 'bytes':
            return send_file(path, mimetype=mimetype, conditional=True)

        start_s, _, end_s = range_spec.partition('-')
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
        end = min(end, file_size - 1)
        if start > end or start >= file_size:
            abort(416)

        length = end - start + 1
        with open(path, 'rb') as f:
            f.seek(start)
            data = f.read(length)

        rv = Response(data, 206, mimetype=mimetype, direct_passthrough=True)
        rv.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
        rv.headers.add('Accept-Ranges', 'bytes')
        rv.headers.add('Content-Length', str(length))
        return rv

    # Serve recordings and face crops
    @app.route('/recordings/<path:filepath>')
    def serve_recording(filepath):
        path = os.path.join(config.RECORDINGS_DIR, filepath)
        return _send_file_partial(path, mimetype='video/mp4')

    @app.route('/face_crops/<path:filepath>')
    def serve_face_crop(filepath):
        return send_file(os.path.join(config.FACE_CROPS_DIR, filepath))

    @app.route('/live')
    def live_stream():
        def generate():
            while True:
                if app.pipeline is None:
                    time.sleep(0.2)
                    continue
                frame = app.pipeline.latest_frame_jpeg
                if frame is None:
                    time.sleep(0.05)
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.1)
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/')
    def index():
        return render_template('index.html')

    return app
