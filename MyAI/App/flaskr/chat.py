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
            # Có thể kiểm tra expert join, thực tế cần phân biệt
            expert_id = session.get('expert_id')
        
        db = get_db()
        # Tìm session_id (giả sử mỗi user chỉ có 1 session active)
        if user_id:
            cs = db.execute('SELECT id FROM chat_session WHERE user_id=? AND status="active"', (user_id,)).fetchone()
        else:
            cs = db.execute('SELECT id FROM chat_session WHERE expert_id=? AND status="active"', (expert_id,)).fetchone()

        session_id = cs['id'] if cs else None

        # Lấy lịch sử tin nhắn từ DB
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
            # Chuyển sang dict để emit qua socket
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
        if user_id:
            cs = db.execute('SELECT id FROM chat_session WHERE user_id=? AND status="active"', (user_id,)).fetchone()
        else:
            cs = db.execute('SELECT id FROM chat_session WHERE expert_id=? AND status="active"', (expert_id,)).fetchone()
        session_id = cs['id'] if cs else None

        if not session_id:
            # Nếu chưa có session_id thì bỏ qua
            return

        content = data.get('message')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Lưu tin nhắn vào DB
        db.execute(
            'INSERT INTO chat_message (session_id, sender_type, sender_id, content, timestamp) VALUES (?, ?, ?, ?, ?)',
            (session_id, sender_type, sender_id, content, timestamp)
        )
        db.execute(
            'UPDATE chat_session SET last_message_at=? WHERE id=?', (timestamp, session_id)
        )
        db.commit()

        # Lấy tên người gửi
        if sender_type == 'user':
            sender_name = db.execute('SELECT username FROM user WHERE id=?', (sender_id,)).fetchone()['username']
        else:
            sender_name = db.execute('SELECT name FROM experts WHERE id=?', (sender_id,)).fetchone()['name']

        msg = {
            "content": content,
            "sender_type": sender_type,
            "sender_name": sender_name,
            "timestamp": timestamp,
        }
        emit('receive_message', msg, broadcast=True)