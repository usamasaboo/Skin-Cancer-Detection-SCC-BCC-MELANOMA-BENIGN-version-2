import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'history.db')

def init_db():
    """Initializes the database and creates necessary tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'patient',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Check if analysis_history exists and needs user_id column
    cursor.execute("PRAGMA table_info(analysis_history)")
    columns = [info[1] for info in cursor.fetchall()]
    
    if 'user_id' not in columns and len(columns) > 0:
        # We need to migrate the old table or just drop it.
        # For this overhaul, we'll recreate it explicitly if needed, but dropping is harsh.
        # Let's add the column via ALTER TABLE if the table exists but is missing the column.
        cursor.execute('ALTER TABLE analysis_history ADD COLUMN user_id INTEGER')
    
    # Ensure Analysis History Table exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            probabilities_json TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Check for missing probabilities_json column
    cursor.execute("PRAGMA table_info(analysis_history)")
    columns = [info[1] for info in cursor.fetchall()]
    if 'probabilities_json' not in columns:
        cursor.execute('ALTER TABLE analysis_history ADD COLUMN probabilities_json TEXT')
    
    # Wellness Logs Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS wellness_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT NOT NULL,
            location TEXT NOT NULL,
            itching BOOLEAN,
            pain BOOLEAN,
            bleeding BOOLEAN,
            size_change TEXT,
            color_change BOOLEAN,
            notes TEXT,
            image_path TEXT,
            cancer_type TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Check for missing cancer_type column
    cursor.execute("PRAGMA table_info(wellness_logs)")
    columns = [info[1] for info in cursor.fetchall()]
    if 'cancer_type' not in columns:
        cursor.execute('ALTER TABLE wellness_logs ADD COLUMN cancer_type TEXT')
    
    # Create a default admin user if none exists
    cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
    if cursor.fetchone()[0] == 0:
        from werkzeug.security import generate_password_hash
        default_admin_pw = generate_password_hash('admin123')
        cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", 
                       ('admin', default_admin_pw, 'admin'))
        print("Default admin created: admin / admin123")
        
    conn.commit()
    conn.close()


import json

def save_analysis(image_path, prediction, confidence, user_id=None, probabilities=None):
    """Saves an analysis record to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    prob_json = json.dumps(probabilities) if probabilities else None
    
    cursor.execute('''
        INSERT INTO analysis_history (user_id, image_path, prediction, confidence, probabilities_json)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, image_path, prediction, float(confidence), prob_json))
    conn.commit()
    conn.close()

def save_wellness_entry(user_id, data):
    """Saves a wellness journal entry."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO wellness_logs (user_id, date, location, itching, pain, bleeding, size_change, color_change, notes, image_path, cancer_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id, 
        data['date'], 
        data['location'], 
        data['itching'], 
        data['pain'], 
        data['bleeding'], 
        data['size_change'], 
        data['color_change'], 
        data['notes'],
        data.get('image_path'),
        data.get('cancer_type')
    ))
    conn.commit()
    conn.close()

def get_wellness_history(user_id):
    """Retrieves wellness journal history for a user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, date, location, itching, pain, bleeding, size_change, color_change, notes, image_path, timestamp, cancer_type
        FROM wellness_logs 
        WHERE user_id = ? 
        ORDER BY date DESC
    ''', (user_id,))
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            'id': row[0],
            'date': row[1],
            'location': row[2],
            'itching': "Yes" if row[3] else "No",
            'pain': "Yes" if row[4] else "No",
            'bleeding': "Yes" if row[5] else "No",
            'size_change': row[6],
            'color_change': "Yes" if row[7] else "No",
            'notes': row[8],
            'image_path': row[9],
            'timestamp': row[10],
            'cancer_type': row[11]
        })
    return history


def get_history(user_id=None, limit=50):
    """Retrieves the analysis history from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if user_id:
        cursor.execute('''
            SELECT id, timestamp, image_path, prediction, confidence 
            FROM analysis_history 
            WHERE user_id = ?
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, limit))
    else:
        cursor.execute('''
            SELECT id, timestamp, image_path, prediction, confidence 
            FROM analysis_history 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
    history = cursor.fetchall()
    conn.close()
    
    # Format for JSON response
    formatted_history = []
    for row in history:
        formatted_history.append({
            'id': row[0],
            'timestamp': row[1],
            'image_name': os.path.basename(row[2]),
            'prediction': row[3],
            'confidence': f"{row[4] * 100:.2f}%"
        })
    return formatted_history

def get_report_by_id(report_id, user_id=None, role='patient'):
    """Retrieves a specific report, verifying access rights."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cols = 'id, user_id, timestamp, image_path, prediction, confidence, probabilities_json'
    if role == 'admin':
        cursor.execute(f'SELECT {cols} FROM analysis_history WHERE id = ?', (report_id,))
    else:
        cursor.execute(f'SELECT {cols} FROM analysis_history WHERE id = ? AND user_id = ?', (report_id, user_id))
        
    row = cursor.fetchone()
    conn.close()
    
    if not row: return None
    
    return {
        'id': row[0],
        'user_id': row[1],
        'timestamp': row[2],
        'image_path': row[3],
        'image_name': os.path.basename(row[3]),
        'prediction': row[4],
        'confidence': f"{row[5] * 100:.2f}%",
        'probabilities': json.loads(row[6]) if row[6] else None
    }

def get_user_by_username(username):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {'id': row[0], 'username': row[1], 'password_hash': row[2], 'role': row[3]}
    return None

def get_user_by_id(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {'id': row[0], 'username': row[1], 'password_hash': row[2], 'role': row[3]}
    return None

def create_user(username, password_hash, role='patient'):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", 
                       (username, password_hash, role))
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        return None # Username exists

def get_dashboard_stats():
    """Generates analytics for the admin dashboard."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    stats = {
        'total_scans': 0,
        'melanoma_count': 0,
        'bcc_count': 0,
        'scc_count': 0,
        'benign_count': 0,
        'total_users': 0,
        'active_protocols': 0,
        'pending_verifications': 0
    }
    
    cursor.execute("SELECT COUNT(*) FROM analysis_history")
    stats['total_scans'] = cursor.fetchone()[0]
    
    cursor.execute("SELECT prediction, COUNT(*) FROM analysis_history GROUP BY prediction")
    counts = dict(cursor.fetchall())
    
    stats['melanoma_count'] = counts.get('melanoma', 0)
    stats['bcc_count'] = counts.get('bcc', 0)
    stats['scc_count'] = counts.get('scc', 0)
    stats['benign_count'] = counts.get('benign', 0)
    
    cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'patient'")
    stats['total_users'] = cursor.fetchone()[0]
    
    # Active Protocols: Check how many models are available (v1 and v2)
    protocols = 0
    if os.path.exists(os.path.join(os.path.dirname(__file__), "saved_models/efficientnet_b0_skin_cancer_4class.pth")):
        protocols += 1
    if os.path.exists(os.path.join(os.path.dirname(__file__), "saved_models/efficientnet_b0_skin_cancer_4class_v2.pth")):
        protocols += 1
    stats['active_protocols'] = max(protocols, 1) # At least 1
    
    # Pending Verifications: Scans with lower confidence (e.g., < 80%)
    cursor.execute("SELECT COUNT(*) FROM analysis_history WHERE confidence < 0.8")
    stats['pending_verifications'] = cursor.fetchone()[0]
    
    conn.close()
    return stats

def update_password(user_id, new_password_hash):
    """Updates the password for a specific user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_password_hash, user_id))
    conn.commit()
    conn.close()

def get_recent_wellness_updates(limit=5):
    """Retrieves the latest wellness journal entries across all patients."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT w.*, u.username 
        FROM wellness_logs w
        JOIN users u ON w.user_id = u.id
        ORDER BY w.timestamp DESC
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_patient_registry():
    """Retrieves a summary of all patients for the registry."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get all patients and their latest scan info
    cursor.execute('''
        SELECT 
            u.id, 
            u.username, 
            u.created_at,
            (SELECT COUNT(*) FROM analysis_history WHERE user_id = u.id) as scan_count,
            (SELECT prediction FROM analysis_history WHERE user_id = u.id ORDER BY timestamp DESC LIMIT 1) as last_prediction,
            (SELECT confidence FROM analysis_history WHERE user_id = u.id ORDER BY timestamp DESC LIMIT 1) as last_confidence,
            (SELECT timestamp FROM analysis_history WHERE user_id = u.id ORDER BY timestamp DESC LIMIT 1) as last_scan_date,
            (SELECT notes FROM wellness_logs WHERE user_id = u.id ORDER BY date DESC LIMIT 1) as last_note,
            (SELECT location FROM wellness_logs WHERE user_id = u.id ORDER BY date DESC LIMIT 1) as last_concern
        FROM users u
        WHERE u.role = 'patient'
        ORDER BY u.created_at DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    patients = []
    for row in rows:
        patients.append(dict(row))
    return patients

def get_patient_detailed_history(user_id):
    """Retrieves full clinical history for a specific patient."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get wellness history
    cursor.execute('''
        SELECT id, date, location, itching, pain, bleeding, size_change, color_change, notes, image_path, timestamp, cancer_type
        FROM wellness_logs 
        WHERE user_id = ? 
        ORDER BY date DESC
    ''', (user_id,))
    wellness = [dict(r) for r in cursor.fetchall()]
    
    # Get scan history
    cursor.execute('''
        SELECT id, timestamp, prediction, confidence, probabilities_json
        FROM analysis_history 
        WHERE user_id = ?
        ORDER BY timestamp DESC
    ''', (user_id,))
    scans = [dict(r) for r in cursor.fetchall()]
    
    conn.close()
    return {'wellness': wellness, 'scans': scans}

def delete_wellness_entry(entry_id, user_id=None, role='patient'):
    """Deletes a specific wellness journal entry, verifying ownership if not admin."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if role == 'admin':
        cursor.execute("DELETE FROM wellness_logs WHERE id = ?", (entry_id,))
    else:
        cursor.execute("DELETE FROM wellness_logs WHERE id = ? AND user_id = ?", (entry_id, user_id))
    conn.commit()
    conn.close()

def delete_analysis_record(record_id, user_id=None, role='patient'):
    """Deletes a specific analysis record, verifying ownership if not admin."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if role == 'admin':
        cursor.execute("DELETE FROM analysis_history WHERE id = ?", (record_id,))
    else:
        cursor.execute("DELETE FROM analysis_history WHERE id = ? AND user_id = ?", (record_id, user_id))
    conn.commit()
    conn.close()

def clear_user_history(user_id):
    """Deletes all analysis history for a specific user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM analysis_history WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

def clear_all_history_global():
    """Wipes all analysis history from the system (Admin only)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM analysis_history")
    conn.commit()
    conn.close()

def export_registry_to_csv():
    """Generates a CSV string of the patient registry."""
    import csv
    import io
    
    patients = get_patient_registry()
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        'id', 'username', 'created_at', 'scan_count', 
        'last_prediction', 'last_confidence', 'last_scan_date', 
        'last_note', 'last_concern'
    ])
    
    writer.writeheader()
    for p in patients:
        writer.writerow(p)
    
    return output.getvalue()

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
