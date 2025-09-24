from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from flaskr.db import get_db

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