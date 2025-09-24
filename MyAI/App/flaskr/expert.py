from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from flaskr.db import get_db
from flaskr.chat import socketio

bp = Blueprint('expert', __name__, url_prefix='/expert')

def expert_required(view):
    from functools import wraps
    def wrapped_view(**kwargs):
        if not session.get('is_expert'):
            return redirect(url_for('auth.login_expert'))
        return view(**kwargs)
    return wraps(view)(wrapped_view)

@bp.route('/questions')
@expert_required
def questions():
    db = get_db()
    # Lấy tất cả câu hỏi, phân trang nếu cần
    questions = db.execute(
        'SELECT * FROM questions ORDER BY created_at DESC'
    ).fetchall()
    return render_template('expert/questions.html', questions=questions)

@bp.route('/questions/<int:question_id>', methods=['GET', 'POST'])
@expert_required
def question_detail(question_id):
    db = get_db()
    question = db.execute(
        'SELECT * FROM questions WHERE id = ?', (question_id,)
    ).fetchone()

    if not question:
        flash('Câu hỏi không tồn tại.', 'error')
        return redirect(url_for('expert.questions'))

    if request.method == 'POST':
        response = request.form['response']
        expert_id = session.get('expert_id')
        db.execute(
            'UPDATE questions SET response=?, status="answered", expert_id=?, responded_at=CURRENT_TIMESTAMP WHERE id=?',
            (response, expert_id, question_id)
        )
        db.commit()
        flash('Đã trả lời câu hỏi thành công.', 'success')
        # TODO: Gửi email phản hồi cho user nếu muốn
        return redirect(url_for('expert.questions'))

    return render_template('expert/question_detail.html', question=question)

@bp.route('/chat-sessions')
@expert_required
def chat_sessions():
    db = get_db()
    expert_id = session.get('expert_id')
    # Lấy danh sách các phiên chat của chuyên gia, mới nhất lên đầu
    sessions = db.execute(
        '''
        SELECT cs.id, cs.started_at, cs.last_message_at, cs.status, u.username, u.email
        FROM chat_session cs
        JOIN user u ON cs.user_id = u.id
        WHERE cs.expert_id = ?
        ORDER BY cs.last_message_at DESC
        ''',
        (expert_id,)
    ).fetchall()
    return render_template('expert/chat_sessions.html', sessions=sessions)

@bp.route('/chat-session/<int:session_id>', methods=['GET', 'POST'])
@expert_required
def chat_session_detail(session_id):
    db = get_db()
    expert_id = session.get('expert_id')
    # Kiểm tra quyền truy cập phiên chat
    session_info = db.execute(
        '''
        SELECT cs.*, u.username, u.email FROM chat_session cs
        JOIN user u ON cs.user_id = u.id
        WHERE cs.id = ? AND cs.expert_id = ?
        ''',
        (session_id, expert_id)
    ).fetchone()
    if not session_info:
        flash('Chat session not found or access denied.', 'error')
        return redirect(url_for('expert.chat_sessions'))

    # Lấy lịch sử tin nhắn
    messages = db.execute(
        '''
        SELECT cm.*, 
            CASE WHEN cm.sender_type='user' THEN u.username ELSE e.name END as sender_name
        FROM chat_message cm
        LEFT JOIN user u ON cm.sender_type='user' AND cm.sender_id=u.id
        LEFT JOIN experts e ON cm.sender_type='expert' AND cm.sender_id=e.id
        WHERE cm.session_id = ?
        ORDER BY cm.timestamp ASC
        ''',
        (session_id,)
    ).fetchall()

    if request.method == 'POST':
        # Gửi tin nhắn mới
        content = request.form.get('message')
        if content:
            db.execute(
                '''
                INSERT INTO chat_message (session_id, sender_type, sender_id, content)
                VALUES (?, 'expert', ?, ?)
                ''',
                (session_id, expert_id, content)
            )
            db.execute(
                'UPDATE chat_session SET last_message_at=CURRENT_TIMESTAMP WHERE id=?', (session_id,)
            )
            db.commit()
            return redirect(url_for('expert.chat_session_detail', session_id=session_id))

    return render_template('expert/expert_realtime_chat.html', session=session_info, messages=messages)