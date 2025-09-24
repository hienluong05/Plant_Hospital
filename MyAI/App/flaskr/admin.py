from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from flaskr.db import get_db

bp = Blueprint('admin', __name__, url_prefix='/admin')

def admin_required(view):
    # Đơn giản: user_id=1 là admin
    from functools import wraps
    def wrapped_view(**kwargs):
        if not session.get('user_id') == 1:
            return redirect(url_for('auth.login'))
        return view(**kwargs)
    return wraps(view)(wrapped_view)

@bp.route('/dashboard')
@admin_required
def dashboard():
    db = get_db()
    total_questions = db.execute('SELECT COUNT(*) FROM questions').fetchone()[0]
    total_consultations = db.execute('SELECT COUNT(*) FROM chat_session').fetchone()[0]
    top_experts = db.execute('SELECT name, rating FROM experts ORDER BY rating DESC LIMIT 5').fetchall()
    completed_visits = db.execute('SELECT COUNT(*) FROM visits WHERE status="completed"').fetchone()[0]
    total_orders = db.execute('SELECT COUNT(*) FROM orders').fetchone()[0]
    return render_template('admin/dashboard.html',
        total_questions=total_questions,
        total_consultations=total_consultations,
        top_experts=top_experts,
        completed_visits=completed_visits,
        total_orders=total_orders
    )

@bp.route('/products')
@admin_required
def products():
    db = get_db()
    products = db.execute('SELECT * FROM products').fetchall()
    return render_template('admin/products.html', products=products)

@bp.route('/products/create', methods=['GET', 'POST'])
@admin_required
def create_product():
    db = get_db()
    if request.method == 'POST':
        name = request.form['name']
        category = request.form.get('category')
        price = request.form['price']
        stock = request.form['stock']
        description = request.form.get('description', '')
        icon = request.form.get('icon', '')
        db.execute(
            'INSERT INTO products (name, category, price, stock, description, icon) VALUES (?, ?, ?, ?, ?, ?)',
            (name, category, price, stock, description, icon)
        )
        db.commit()
        flash('Product added!', 'success')
        return redirect(url_for('admin.products'))
    return render_template('admin/product_form.html', product=None)

@bp.route('/products/edit/<int:product_id>', methods=['GET', 'POST'])
@admin_required
def edit_product(product_id):
    db = get_db()
    product = db.execute('SELECT * FROM products WHERE id=?', (product_id,)).fetchone()
    if not product:
        flash('Product not found.', 'error')
        return redirect(url_for('admin.products'))
    if request.method == 'POST':
        db.execute(
            'UPDATE products SET name=?, category=?, price=?, stock=?, description=?, icon=?, updated_at=CURRENT_TIMESTAMP WHERE id=?',
            (request.form['name'], request.form['category'], request.form['price'],
             request.form['stock'], request.form['description'], request.form['icon'], product_id)
        )
        db.commit()
        flash('Product updated.', 'success')
        return redirect(url_for('admin.products'))
    return render_template('admin/product_form.html', product=product)

@bp.route('/products/delete/<int:product_id>', methods=['POST'])
@admin_required
def delete_product(product_id):
    db = get_db()
    db.execute('DELETE FROM products WHERE id=?', (product_id,))
    db.commit()
    flash('Product deleted.', 'success')
    return redirect(url_for('admin.products'))

@bp.route('/orders')
@admin_required
def orders():
    db = get_db()
    orders = db.execute('SELECT * FROM orders ORDER BY created_at DESC').fetchall()
    return render_template('admin/orders.html', orders=orders)

@bp.route('/orders/<int:order_id>')
@admin_required
def order_detail(order_id):
    db = get_db()
    order = db.execute('SELECT * FROM orders WHERE id=?', (order_id,)).fetchone()
    items = db.execute('SELECT * FROM order_items WHERE order_id=?', (order_id,)).fetchall()
    return render_template('admin/order_detail.html', order=order, items=items)

@bp.route('/consultations')
@admin_required
def consultations():
    db = get_db()
    sessions = db.execute('''
        SELECT chat_session.*, user.username as user_name, experts.name as expert_name
        FROM chat_session
        LEFT JOIN user ON chat_session.user_id = user.id
        LEFT JOIN experts ON chat_session.expert_id = experts.id
        ORDER BY chat_session.started_at DESC
    ''').fetchall()
    return render_template('admin/consultations.html', sessions=sessions)

@bp.route('/consultations/<int:session_id>')
@admin_required
def consultation_detail(session_id):
    db = get_db()
    session_row = db.execute(
        '''SELECT chat_session.*, user.username as user_name, experts.name as expert_name
           FROM chat_session
           LEFT JOIN user ON chat_session.user_id = user.id
           LEFT JOIN experts ON chat_session.expert_id = experts.id
           WHERE chat_session.id=?''', (session_id,)
    ).fetchone()
    if not session_row:
        flash('Session not found.', 'error')
        return redirect(url_for('admin.consultations'))
    messages = db.execute(
        '''SELECT * FROM chat_message WHERE session_id=? ORDER BY timestamp ASC''', (session_id,)
    ).fetchall()
    return render_template('admin/consultation_detail.html', session=session_row, messages=messages)