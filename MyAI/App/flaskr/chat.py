from flask_socketio import SocketIO, emit, join_room
from flask import session, g
from flaskr.db import get_db
from datetime import datetime

socketio = None

def init_socketio(app):
    global socketio
    socketio = SocketIO(app)

    @socketio.on('join_chat')
    def handle_join_chat():
        user_id = session.get('user_id')
        expert_id = None
        if not user_id:
            expert_id = session.get('expert_id')

        db = get_db()
        session_id = None

        # Tìm hoặc tạo session cho user mới
        if user_id:
            cs = db.execute('SELECT id FROM chat_session WHERE user_id=? AND status="active"', (user_id,)).fetchone()
            if not cs:
                # Chọn expert bất kỳ (hoặc logic khác)
                expert = db.execute('SELECT id FROM experts WHERE active=1 LIMIT 1').fetchone()
                if expert:
                    db.execute(
                        'INSERT INTO chat_session (user_id, expert_id, status) VALUES (?, ?, "active")',
                        (user_id, expert['id'])
                    )
                    db.commit()
                    cs = db.execute('SELECT id FROM chat_session WHERE user_id=? AND status="active"', (user_id,)).fetchone()
            session_id = cs['id'] if cs else None
        elif expert_id:
            cs = db.execute('SELECT id FROM chat_session WHERE expert_id=? AND status="active"', (expert_id,)).fetchone()
            session_id = cs['id'] if cs else None

        # Lấy lịch sử tin nhắn từ DB như cũ
        messages = []
        if session_id:
            messages = db.execute(
                '''SELECT cm.*, 
                        CASE WHEN cm.sender_type='user' THEN u.username ELSE e.name END as sender_name
                FROM chat_message cm
                LEFT JOIN user u ON cm.sender_type='user' AND cm.sender_id=u.id
                LEFT JOIN experts e ON cm.sender_type='expert' AND cm.sender_id=e.id
                WHERE cm.session_id = ?
                ORDER BY cm.timestamp ASC''', (session_id,)
            ).fetchall()
            messages = [
                {
                    'content': m['content'],
                    'sender_type': m['sender_type'],
                    'sender_name': m['sender_name'],
                    'timestamp': m['timestamp'],
                } for m in messages
            ]
        emit('chat_history', messages)

    @socketio.on('send_message')
    def handle_send_message(data):
        db = get_db()
        user_id = session.get('user_id')
        expert_id = session.get('expert_id')
        sender_type = 'user' if user_id else 'expert'
        sender_id = user_id or expert_id

        # Xác định session_id
        session_id = None
        if user_id:
            cs = db.execute('SELECT id FROM chat_session WHERE user_id=? AND status="active"', (user_id,)).fetchone()
            if not cs:
                # Chọn expert bất kỳ, hoặc logic khác
                expert = db.execute('SELECT id FROM experts WHERE active=1 LIMIT 1').fetchone()
                if not expert:
                    # Không có chuyên gia nào
                    return
                db.execute(
                    'INSERT INTO chat_session (user_id, expert_id, status) VALUES (?, ?, "active")',
                    (user_id, expert['id'])
                )
                db.commit()
                cs = db.execute('SELECT id FROM chat_session WHERE user_id=? AND status="active"', (user_id,)).fetchone()
            session_id = cs['id'] if cs else None
        else:
            cs = db.execute('SELECT id FROM chat_session WHERE expert_id=? AND status="active"', (expert_id,)).fetchone()
            session_id = cs['id'] if cs else None

        if not session_id:
            return

        content = data.get('message')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        db.execute(
            'INSERT INTO chat_message (session_id, sender_type, sender_id, content, timestamp) VALUES (?, ?, ?, ?, ?)',
            (session_id, sender_type, sender_id, content, timestamp)
        )
        db.execute(
            'UPDATE chat_session SET last_message_at=? WHERE id=?', (timestamp, session_id)
        )
        db.commit()

        if sender_type == 'user':
            sender_row = db.execute('SELECT username FROM user WHERE id=?', (sender_id,)).fetchone()
            sender_name = sender_row['username'] if sender_row else 'User'
        else:
            sender_row = db.execute('SELECT name FROM experts WHERE id=?', (sender_id,)).fetchone()
            sender_name = sender_row['name'] if sender_row else 'Expert'

        msg = {
            "content": content,
            "sender_type": sender_type,
            "sender_name": sender_name,
            "timestamp": timestamp,
        }
        emit('receive_message', msg, broadcast=True)