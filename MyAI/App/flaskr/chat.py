from flask_socketio import SocketIO, emit, join_room
from flask import session, g, copy_current_request_context
from flaskr.db import get_db
from datetime import datetime

socketio = None  # Sẽ được khởi tạo trong __init__.py

messages = []  # Lưu lịch sử chat cho demo, thực tế nên lưu DB

def init_socketio(app):
    global socketio
    socketio = SocketIO(app)

    @socketio.on('join_chat')
    def handle_join_chat():
        emit('chat_history', messages)

    @socketio.on('send_message')
    def handle_send_message(data):
        msg = {
            "user_name": session.get('username', 'User'),
            "message": data['message'],
            "is_expert": session.get('is_expert', False),
        }
        messages.append(msg)
        emit('receive_message', msg, broadcast=True)