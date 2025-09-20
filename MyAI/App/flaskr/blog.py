from flask import Blueprint, render_template, request, redirect, url_for, session, g, flash
from flaskr.db import get_db
from flaskr.auth import login_required  # nếu có, dùng để bảo vệ route
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

@bp.route('/my-blogs')
def my_blogs():
    if g.user is None:
        return redirect(url_for('auth.login'))
    db = get_db()
    posts = db.execute(
        'SELECT p.id, title, body, created '
        'FROM post p '
        'WHERE p.author_id = ? '
        'ORDER BY created DESC',
        (g.user['id'],)
    ).fetchall()
    return render_template('blog/my_blogs.html', posts=posts)

@bp.route('/edit/<int:id>', methods=('GET', 'POST'))
def edit(id):
    db = get_db()
    post = db.execute(
        'SELECT * FROM post WHERE id = ? AND author_id = ?', (id, g.user['id'])
    ).fetchone()
    if post is None:
        return "Not found or no permission", 404

    if request.method == 'POST':
        title = request.form['title']
        body = request.form['body']
        error = None
        if not title:
            error = "Title is required."
        if error:
            flash(error)
        else:
            db.execute(
                'UPDATE post SET title = ?, body = ? WHERE id = ?',
                (title, body, id)
            )
            db.commit()
            return redirect(url_for('blog.my_blogs'))
    return render_template('blog/edit.html', post=post)

@bp.route('/delete/<int:id>', methods=['POST'])
def delete(id):
    db = get_db()
    post = db.execute(
        'SELECT * FROM post WHERE id = ? AND author_id = ?', (id, g.user['id'])
    ).fetchone()
    if post is None:
        return "Not found or no permission", 404
    db.execute('DELETE FROM post WHERE id = ?', (id,))
    db.commit()
    flash('Deleted successfully!')
    return redirect(url_for('blog.my_blogs'))