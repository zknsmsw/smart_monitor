"""SQLite database for recordings, face groups, face events, motion events."""
import sqlite3
import threading
import logging
from datetime import datetime

log = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        self._local = threading.local()
        self._init_schema()

    @property
    def _conn(self):
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute('PRAGMA journal_mode=WAL')
        return self._local.conn

    def _init_schema(self):
        c = self._conn
        c.executescript('''
            CREATE TABLE IF NOT EXISTS recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                duration REAL,
                file_size INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_recordings_start ON recordings(start_time);

            CREATE TABLE IF NOT EXISTS face_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT DEFAULT '',
                representative_crop TEXT,
                mean_embedding BLOB,
                face_count INTEGER DEFAULT 1,
                first_seen TEXT,
                last_seen TEXT
            );

            CREATE TABLE IF NOT EXISTS face_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_group_id INTEGER REFERENCES face_groups(id),
                recording_id INTEGER REFERENCES recordings(id),
                timestamp TEXT NOT NULL,
                bbox TEXT,
                confidence REAL,
                crop_path TEXT,
                embedding BLOB
            );
            CREATE INDEX IF NOT EXISTS idx_face_events_group ON face_events(face_group_id);
            CREATE INDEX IF NOT EXISTS idx_face_events_time ON face_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_face_events_rec ON face_events(recording_id);

            CREATE TABLE IF NOT EXISTS motion_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recording_id INTEGER REFERENCES recordings(id),
                start_time TEXT NOT NULL,
                end_time TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_motion_start ON motion_events(start_time);
        ''')
        c.commit()

    # --- Recordings ---
    def add_recording(self, file_path, start_time, end_time, duration, file_size):
        c = self._conn
        cur = c.execute(
            'INSERT INTO recordings (file_path, start_time, end_time, duration, file_size) VALUES (?,?,?,?,?)',
            (file_path, start_time, end_time, duration, file_size))
        c.commit()
        return cur.lastrowid

    def get_recordings_by_date(self, date_str):
        return self._conn.execute(
            "SELECT * FROM recordings WHERE start_time LIKE ? ORDER BY start_time",
            (f'{date_str}%',)).fetchall()

    def get_recording(self, rec_id):
        return self._conn.execute('SELECT * FROM recordings WHERE id=?', (rec_id,)).fetchone()

    def get_oldest_recordings(self, limit=10):
        return self._conn.execute(
            'SELECT * FROM recordings ORDER BY start_time ASC LIMIT ?', (limit,)).fetchall()

    def delete_recording(self, rec_id):
        c = self._conn
        c.execute('DELETE FROM face_events WHERE recording_id=?', (rec_id,))
        c.execute('DELETE FROM motion_events WHERE recording_id=?', (rec_id,))
        c.execute('DELETE FROM recordings WHERE id=?', (rec_id,))
        c.commit()

    # --- Face Groups ---
    def add_face_group(self, representative_crop, mean_embedding):
        now = datetime.now().isoformat()
        c = self._conn
        cur = c.execute(
            'INSERT INTO face_groups (representative_crop, mean_embedding, face_count, first_seen, last_seen) VALUES (?,?,1,?,?)',
            (representative_crop, mean_embedding, now, now))
        c.commit()
        return cur.lastrowid

    def get_all_face_groups(self):
        return self._conn.execute('SELECT * FROM face_groups ORDER BY last_seen DESC').fetchall()

    def get_face_group(self, group_id):
        return self._conn.execute('SELECT * FROM face_groups WHERE id=?', (group_id,)).fetchone()

    def get_face_group_count(self, group_id):
        row = self._conn.execute('SELECT face_count FROM face_groups WHERE id=?', (group_id,)).fetchone()
        return row['face_count'] if row else 0

    def update_face_group_embedding(self, group_id, mean_embedding, face_count):
        now = datetime.now().isoformat()
        self._conn.execute(
            'UPDATE face_groups SET mean_embedding=?, face_count=?, last_seen=? WHERE id=?',
            (mean_embedding, face_count, now, group_id))
        self._conn.commit()

    def update_face_group_crop(self, group_id, crop_path):
        self._conn.execute('UPDATE face_groups SET representative_crop=? WHERE id=?', (crop_path, group_id))
        self._conn.commit()

    def update_face_group_label(self, group_id, label):
        self._conn.execute('UPDATE face_groups SET label=? WHERE id=?', (label, group_id))
        self._conn.commit()

    # --- Face Events ---
    def add_face_event(self, face_group_id, recording_id, bbox, confidence, crop_path, embedding):
        now = datetime.now().isoformat()
        bbox_str = ','.join(map(str, bbox)) if bbox else ''
        c = self._conn
        cur = c.execute(
            'INSERT INTO face_events (face_group_id, recording_id, timestamp, bbox, confidence, crop_path, embedding) VALUES (?,?,?,?,?,?,?)',
            (face_group_id, recording_id, now, bbox_str, confidence, crop_path, embedding))
        c.commit()
        return cur.lastrowid

    def update_face_event_crop(self, event_id, crop_path):
        self._conn.execute('UPDATE face_events SET crop_path=? WHERE id=?', (crop_path, event_id))
        self._conn.commit()

    def get_face_events_by_group(self, group_id):
        return self._conn.execute(
            'SELECT * FROM face_events WHERE face_group_id=? ORDER BY timestamp DESC',
            (group_id,)).fetchall()

    def get_face_events_by_recording(self, recording_id):
        return self._conn.execute(
            'SELECT fe.*, fg.label FROM face_events fe LEFT JOIN face_groups fg ON fe.face_group_id=fg.id WHERE fe.recording_id=? ORDER BY fe.timestamp',
            (recording_id,)).fetchall()

    # --- Motion Events ---
    def add_motion_event(self, recording_id, start_time):
        c = self._conn
        cur = c.execute('INSERT INTO motion_events (recording_id, start_time) VALUES (?,?)',
                        (recording_id, start_time))
        c.commit()
        return cur.lastrowid

    def end_motion_event(self, event_id):
        now = datetime.now().isoformat()
        self._conn.execute('UPDATE motion_events SET end_time=? WHERE id=?', (now, event_id))
        self._conn.commit()

    def get_motion_events_by_date(self, date_str):
        return self._conn.execute(
            "SELECT * FROM motion_events WHERE start_time LIKE ? ORDER BY start_time",
            (f'{date_str}%',)).fetchall()

    def get_timeline(self, date_str):
        """Get combined timeline of motion + face events for a date."""
        motions = [dict(r) for r in self.get_motion_events_by_date(date_str)]
        faces = self._conn.execute(
            "SELECT fe.timestamp, fe.face_group_id, fg.label, fe.crop_path, fe.confidence "
            "FROM face_events fe LEFT JOIN face_groups fg ON fe.face_group_id=fg.id "
            "WHERE fe.timestamp LIKE ? ORDER BY fe.timestamp",
            (f'{date_str}%',)).fetchall()
        return {'motions': motions, 'faces': [dict(f) for f in faces]}

    # --- Status ---
    def get_stats(self):
        c = self._conn
        rec_count = c.execute('SELECT COUNT(*) FROM recordings').fetchone()[0]
        face_count = c.execute('SELECT COUNT(*) FROM face_groups').fetchone()[0]
        event_count = c.execute('SELECT COUNT(*) FROM face_events').fetchone()[0]
        return {'recordings': rec_count, 'face_groups': face_count, 'face_events': event_count}
