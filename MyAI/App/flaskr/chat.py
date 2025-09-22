from flask_socketio import SocketIO, emit, join_room
from flask import session, g, copy_current_request_context
from flaskr.db import get_db
from datetime import datetime

socketio = None

def init_socketio(app):
    global socketio
    socketio = SocketIO(app, manage_session=False)

    @socketio.on('get_expert_sessions')
    def get_expert_sessions(data):
        expert_id = session.get('expert_id')
        if not expert_id:
            emit('expert_sessions', [])
            return
        db = get_db()
        sessions = db.execute(
            """SELECT cs.id as session_id, u.username as user_name, u.id as user_id, u.username, u.email, u.id,
                      (SELECT content FROM chat_message WHERE session_id=cs.id ORDER BY timestamp DESC LIMIT 1) as last_message,
                      '' as avatar
                FROM chat_session cs
                JOIN user u ON cs.user_id = u.id
                WHERE cs.expert_id = ?
                ORDER BY cs.last_message_at DESC
            """, (expert_id,)
        ).fetchall()
        # Chuyển sang dict đơn giản
        result = []
        for row in sessions:
            result.append({
                "session_id": row["session_id"],
                "user_name": row["user_name"],
                "last_message": row["last_message"] or "",
                "avatar": row["avatar"] or "",
            })
        emit('expert_sessions', result)

    @socketio.on('get_chat_history')
    def get_chat_history(data):
        session_id = data.get('session_id')
        db = get_db()
        msgs = db.execute(
            "SELECT sender_type, content, timestamp FROM chat_message WHERE session_id=? ORDER BY timestamp ASC", 
            (session_id,)
        ).fetchall()
        messages = [{"sender_type": m["sender_type"], "content": m["content"], "timestamp": m["timestamp"]} for m in msgs]
        emit('chat_history', messages)

    @socketio.on('send_message')
    def send_message(data):
        expert_id = session.get('expert_id')
        if not expert_id:
            return
        session_id = data.get('session_id')
        content = data.get('content')
        sender_type = data.get('sender_type', 'expert')
        db = get_db()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        db.execute(
            "INSERT INTO chat_message (session_id, sender_type, sender_id, content, timestamp) VALUES (?, ?, ?, ?, ?)",
            (session_id, sender_type, expert_id, content, now)
        )
        db.execute(
            "UPDATE chat_session SET last_message_at=? WHERE id=?",
            (now, session_id)
        )
        db.commit()
        msg = {
            "session_id": session_id,
            "sender_type": sender_type,
            "content": content,
            "timestamp": now
        }
        emit('receive_message', msg, broadcast=True)