from flask import Blueprint, render_template, request, redirect, url_for, session, g, flash
from flaskr.db import get_db

bp = Blueprint('blog', __name__, url_prefix='/blog')

@bp.route('/')
def index():
    db = get_db()
    posts = db.execute(
        'SELECT p.id, title, body, created, username'
        ' FROM post p JOIN user u ON p.author_id = u.id'
        ' ORDER BY created DESC'
    ).fetchall()
    return render_template('blog/home.html', posts=posts)

@bp.route('/create', methods=('GET', 'POST'))
def create():
    if g.user is None:
        return redirect(url_for('auth.login'))
    if request.method == 'POST':
        title = request.form['title']
        body = request.form['body']
        error = None
        if not title:
            error = 'Title is required.'
        if error is not None:
            flash(error)
        else:
            db = get_db()
            db.execute(
                'INSERT INTO post (title, body, author_id) VALUES (?, ?, ?)',
                (title, body, g.user['id'])
            )
            db.commit()
            return redirect(url_for('blog.index'))
    return render_template('blog/create.html')

@bp.route('/<int:id>')
def detail(id):
    db = get_db()
    post = db.execute(
        'SELECT p.id, title, body, created, username'
        ' FROM post p JOIN user u ON p.author_id = u.id'
        ' WHERE p.id = ?', (id,)
    ).fetchone()
    if post is None:
        return "Not found", 404
    return render_template('blog/detail.html', post=post)