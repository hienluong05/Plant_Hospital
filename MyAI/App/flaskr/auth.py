import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

from werkzeug.security import check_password_hash, generate_password_hash

from flaskr.db import get_db

bp = Blueprint('auth', __name__, url_prefix = '/auth')

@bp.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        db = get_db()
        error = None

        if not username:
            error = 'Tên người dùng là bắt buộc.'
        elif not email:
            error = 'Email là bắt buộc.'
        elif not password:
            error = 'Mật khẩu là bắt buộc.'
        elif password != confirm_password:
            error = 'Mật khẩu xác nhận không khớp.'
        elif len(password) < 6:
            error = 'Mật khẩu phải có ít nhất 6 ký tự.'

        if error is None:
            try:
                db.execute(
                    "INSERT INTO user (username, email, password) VALUES (?, ?, ?)",
                    (username, email, generate_password_hash(password)),
                )
                db.commit()
                return redirect(url_for("auth.login", success="Đăng ký thành công! Vui lòng đăng nhập."))
            except db.IntegrityError:
                error = f"Email {email} đã được sử dụng."

        return render_template('auth/register.html', error=error)

    return render_template('auth/register.html')

@bp.route('/login', methods=('GET', 'POST'))
def login():
    success_message = request.args.get('success')
    
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        db = get_db()
        error = None
        user = db.execute(
            'SELECT * FROM user WHERE email = ?', (email,)
        ).fetchone()

        if user is None:
            error = 'Email does not exists.'
        elif not check_password_hash(user['password'], password):
            error = 'Password is not correct.'

        if error is None:
            session.clear()
            session['user_id'] = user['id']
            return redirect(url_for('home'))

        return render_template('auth/login.html', error=error)

    return render_template('auth/login.html', success=success_message)

@bp.route('/login-expert', methods=['GET', 'POST'])
def login_expert():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        db = get_db()
        expert = db.execute('SELECT * FROM experts WHERE email = ?', (email,)).fetchone()
        if expert is None or password != '123456':
            error = 'Email hoặc mật khẩu không đúng'
        else:
            session.clear()
            session['expert_id'] = expert['id']
            session['is_expert'] = True
            session['username'] = expert['name']
            return redirect(url_for('expert_dashboard'))
    return render_template('auth/expert_login.html', error=error)  # Đúng với tên file template

@bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = get_db().execute(
            'SELECT * FROM user WHERE id = ?', (user_id,)
        ).fetchone()
        
@bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for('auth.login'))
        return view(**kwargs)
    return wrapped_view