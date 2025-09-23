import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, g, flash
from werkzeug.utils import secure_filename
from datetime import datetime
import pandas as pd
import numpy as np
import os
from torchvision.models import resnet50, ResNet50_Weights

from flask_socketio import SocketIO
from . import chat
from flask import session, jsonify, request
import feedparser
from .vnpay_utils import build_vnpay_url, verify_vnpay_response

def get_class_names_from_folder(dataset_dir):
    class_names = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    class_names.sort()
    return class_names


def get_latest_news(limit_per_source=5):
    sources = [
        {
            "name": "AgWeb",
            "url": "https://www.agweb.com/rss.xml"
        },
        {
            "name": "Successful Farming",
            "url": "https://www.agriculture.com/rss.xml"
        },
        {
            "name": "The Western Producer",
            "url": "https://www.producer.com/feed/"
        },
        {
            "name": "Reuters Agriculture",
            "url": "https://www.reuters.com/rssFeed/agricultureNews"
        },
        {
            "name": "BBC Sci&Env",
            "url": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"
        },
    ]
    news_list = []
    for src in sources:
        feed = feedparser.parse(src["url"])
        for entry in feed.entries[:limit_per_source]:
            news_list.append({
                "title": entry.title,
                "link": entry.link,
                "published": entry.published if 'published' in entry else '',
                "summary": entry.summary if 'summary' in entry else '',
                "source": src["name"]
            })
    # Sắp xếp toàn bộ theo thời gian mới nhất (nếu muốn)
    news_list.sort(key=lambda x: x['published'], reverse=True)
    return news_list

def create_app():
    app = Flask(__name__)

    # Cấu hình app
    app.config.from_mapping(
        SECRET_KEY='your-secret-key-here',
        DATABASE=os.path.join(app.instance_path, 'plant_disease.sqlite'),
    )

    # Cấu hình upload
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

    # Tạo các thư mục cần thiết
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    try:
        os.makedirs(os.path.join(os.path.dirname(__file__), 'static'))
    except OSError:
        pass

    try:
        os.makedirs(UPLOAD_FOLDER)
    except OSError:
        pass

    # Khởi tạo mô hình chỉ 1 lần
    DATASET_DIR = r"C:\PlantVillage"
    disease_names = get_class_names_from_folder(DATASET_DIR)
    print("Tên class sẽ được dùng:", disease_names)
    NUM_CLASSES = 15

    class CNN(nn.Module):
        def __init__(self, num_classes):
            super(CNN, self).__init__()
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        
        def forward(self, x):
            return self.model(x)

    model = CNN(NUM_CLASSES)
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'plant_disease_detection_model.pt'), map_location='cpu'))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Khởi tạo database
    from . import db
    db.init_app(app)

    with app.app_context():
        db_path = app.config['DATABASE']
        if not os.path.exists(db_path):
            print("Database does not exist yet, automatically initialized from schema.sql...")
            db.init_db()
    
    # Đăng ký blueprint auth
    from . import auth
    app.register_blueprint(auth.bp)
    
    # Đăng ký blueprint blog
    from . import blog
    app.register_blueprint(blog.bp)

    # Helper function để get database
    def get_db():
        return db.get_db()
    
    chat.init_socketio(app) 

    # Route cho trang chủ (không cần đăng nhập)
    @app.route('/')
    def home():
        news_list = get_latest_news()
        return render_template('home.html', news_list=news_list)

    # Route cho AI diagnosis (cần đăng nhập)
    @app.route('/ai-diagnosis', methods=['GET', 'POST'])
    def ai_diagnosis():
        # Kiểm tra xem user đã đăng nhập chưa
        if g.user is None:
            return redirect(url_for('auth.login'))
            
        result = None
        filename = None
        if request.method == 'POST':
            if 'file' not in request.files:
                result = "No file uploaded!"
                return render_template('ai_diagnosis.html', result=result)
            file = request.files['file']
            if file.filename == '':
                result = "No file selected!"
                return render_template('ai_diagnosis.html', result=result)
            if file:
                img = Image.open(file.stream).convert("RGB")
                img_tran = transform(img)
                img_tran = img_tran.unsqueeze(0)
                with torch.no_grad():
                    output = model(img_tran)
                    pred_idx = output.argmax(dim=1).item()
                    result = disease_names[pred_idx]
                filename = os.path.join('static', 'upload.jpg')
                img.save(os.path.join(os.path.dirname(__file__), filename))
                return render_template('ai_diagnosis.html', result=result, image_file=filename)
        return render_template('ai_diagnosis.html', result=result)

    # Route cho pharmacy
    @app.route('/pharmacy')
    def pharmacy():
        return render_template('pharmacy.html')

    @app.route('/add-to-cart', methods=['POST'])
    def add_to_cart():
        data = request.json
        if not data or 'id' not in data:
            return jsonify({'status': 'error', 'message': 'No product info'}), 400

        product = {
            'id': data['id'],
            'name': data['name'],
            'price': data['price'],
            'quantity': 1
        }

        # Khởi tạo giỏ hàng nếu chưa có
        if 'cart' not in session:
            session['cart'] = []

        cart = session['cart']
        # Kiểm tra nếu đã có thì tăng số lượng
        for item in cart:
            if item['id'] == product['id']:
                item['quantity'] += 1
                session['cart'] = cart
                return jsonify({'status': 'ok', 'cart': cart})
        # Nếu chưa có thì thêm mới
        cart.append(product)
        session['cart'] = cart
        return jsonify({'status': 'ok', 'cart': cart})

    @app.route('/cart')
    def cart():
        cart = session.get("cart", [])
        total = sum(float(item['price'].replace('$','')) * item['quantity'] for item in cart)
        return render_template("cart.html", cart=cart, total=total)

    @app.route('/remove-from-cart', methods=['POST'])
    def remove_from_cart():
        data = request.get_json()
        product_id = data.get("id")
        cart = session.get("cart", [])
        cart = [item for item in cart if item["id"] != product_id]
        session["cart"] = cart
        return jsonify({"status": "ok"})

    @app.route('/update-cart', methods=['POST'])
    def update_cart():
        data = request.json
        product_id = data.get('id')
        quantity = int(data.get('quantity', 1))
        cart = session.get('cart', [])
        for item in cart:
            if item['id'] == product_id:
                item['quantity'] = quantity
                break
        session['cart'] = cart
        return jsonify({'status': 'ok'})

    # Route cho expert consultation - CHỈ ĐỊNH NGHĨA 1 LẦN
    @app.route('/expert-consultation')
    def expert_consultation():
        return render_template('expert_consultation.html')

    # Route cho submit question
    @app.route('/submit-question', methods=['POST'])
    def submit_question():
        try:
            name = request.form['name']
            email = request.form['email']
            phone = request.form.get('phone', '')
            plant_type = request.form['plant_type']
            question = request.form['question']
            
            # Xử lý upload hình ảnh
            uploaded_files = []
            if 'images' in request.files:
                files = request.files.getlist('images')
                for file in files:
                    if file and file.filename:
                        filename = secure_filename(file.filename)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
                        filename = timestamp + filename
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(file_path)
                        uploaded_files.append(filename)
            
            # Lưu vào database
            database = get_db()
            database.execute(
                'INSERT INTO questions (name, email, phone, plant_type, question, images, created_at, status)'
                ' VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (name, email, phone, plant_type, question, ','.join(uploaded_files), 
                 datetime.now(), 'pending')
            )
            database.commit()
            
            flash('Question sent successfully! We will respond within 24-48 hours.', 'success')
            return redirect(url_for('expert_consultation'))
            
        except Exception as e:
            flash('An error occurred. Please try again later.', 'error')
            return redirect(url_for('expert_consultation'))
    
    # Config vnpay
    app.config['VNPAY_TMN_CODE'] = 'MQ6F4LQX' # Lấy từ VNPay
    app.config['VNPAY_HASH_SECRET'] = 'B5ZAH92LELL58F9544OUQOEQNIH07DJZ'
    app.config['VNPAY_URL'] = 'https://sandbox.vnpayment.vn/paymentv2/vpcpay.html'
    app.config['VNPAY_RETURN_URL'] = 'http://127.0.0.1:5000/vnpay_return'  # Sửa cho phù hợp domain của bạn

    # # Route cho book visit
    # @app.route('/book-visit', methods=['POST'])
    # def book_visit():
    #     try:
    #         name = request.form['visit_name']
    #         email = request.form['visit_email']
    #         phone = request.form['visit_phone']
    #         address = request.form['visit_address']
    #         visit_date = request.form['visit_date']
    #         visit_time = request.form['visit_time']
    #         garden_size = request.form['garden_size']
            
    #         # Lưu vào database
    #         database = get_db()
    #         database.execute(
    #             'INSERT INTO visits (name, email, phone, address, visit_date, visit_time, garden_size, created_at, status)'
    #             ' VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
    #             (name, email, phone, address, visit_date, visit_time, garden_size, 
    #              datetime.now(), 'pending')
    #         )
    #         database.commit()
            
    #         flash('Appointment successful! We will contact you for confirmation as soon as possible.', 'success')
    #         return redirect(url_for('expert_consultation'))
            
    #     except Exception as e:
    #         flash('An error occurred. Please try again later.', 'error')
    #         return redirect(url_for('expert_consultation'))
    
    @app.route('/book-visit-vnpay', methods=['POST'])
    def book_visit_vnpay():
        name = request.form['visit_name']
        email = request.form['visit_email']
        phone = request.form['visit_phone']
        address = request.form['visit_address']
        visit_date = request.form['visit_date']
        visit_time = request.form['visit_time']
        garden_size = request.form['garden_size']

        order_id = 'VISIT' + datetime.now().strftime('%Y%m%d%H%M%S')
        price_vnd = 500000  # VND trực tiếp

        # Lưu visit, nhớ để notes=order_id để UPDATE được khi trả về
        db = get_db()
        db.execute(
            "INSERT INTO visits (name, email, phone, address, visit_date, visit_time, garden_size, status, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (name, email, phone, address, visit_date, visit_time, garden_size, 'pending', order_id)
        )
        db.commit()

        order_desc = f"Home consultation {order_id} for {name}"
        vnp_url = build_vnpay_url(
            order_id=order_id,
            amount_vnd=price_vnd,
            order_desc=order_desc,
            config=app.config,
            user_ip=request.remote_addr
        )
        return redirect(vnp_url)

    # Route cho about
    @app.route('/about')
    def about():
        return render_template('about.html')

    # Route cho contact
    @app.route('/contact')
    def contact():
        return render_template('contact.html')
    
    @app.route('/expert-chat')
    def expert_chat():
        # Ở đây bạn có thể render một trang chat chuyên gia, tạm thời render template demo
        return render_template('expert_chat.html')
    
    from .vnpay_utils import build_vnpay_url

    @app.route('/checkout-vnpay')
    def checkout_vnpay():
        cart = session.get('cart', [])
        if not cart:
            flash('Cart is empty!', 'error')
            return redirect(url_for('cart'))

        # Tổng USD hiện đang lưu dạng "$xx.yy"
        total_usd = sum(float(item['price'].replace('$', '')) * item['quantity'] for item in cart)

        # Đổi sang VND 1 lần (cố định 25,000 hoặc nếu bạn có API thì dùng)
        RATE = 25000
        total_vnd = int(round(total_usd * RATE))

        order_id = 'ORDER' + datetime.now().strftime('%Y%m%d%H%M%S')
        # (Khuyến nghị) Lưu đơn hàng vào DB orders để đối chiếu khi return:
        db = get_db()
        db.execute("INSERT INTO orders (user_id, total, status) VALUES (?, ?, ?)",
                (g.user['id'] if getattr(g, 'user', None) else None,
                    total_usd,
                    'pending'))
        db.commit()

        order_desc = f"Pay order {order_id}"
        vnp_url = build_vnpay_url(
            order_id=order_id,
            amount_vnd=total_vnd,
            order_desc=order_desc,
            config=app.config,
            user_ip=request.remote_addr
        )
        return redirect(vnp_url)
        
    @app.route('/vnpay_return')
    def vnpay_return():
        params = request.args.to_dict()
        if not verify_vnpay_response(params, app.config):
            flash('Invalid signature!', 'error')
            return redirect(url_for('home'))

        vnp_response_code = params.get('vnp_ResponseCode')
        order_id = params.get('vnp_TxnRef')
        db = get_db()

        if order_id and order_id.startswith('VISIT'):
            if vnp_response_code == '00':
                db.execute("UPDATE visits SET status='paid' WHERE notes=?", (order_id,))
                db.commit()
                flash('Payment successful! Booking confirmed.', 'success')
            else:
                flash('Payment failed or canceled.', 'error')
            return redirect(url_for('expert_consultation'))

        if order_id and order_id.startswith('ORDER'):
            if vnp_response_code == '00':
                session['cart'] = []
                flash('Order payment successful!', 'success')
            else:
                flash('Payment failed or canceled.', 'error')
            return redirect(url_for('cart'))

        flash('Order reference not recognized.', 'error')
        return redirect(url_for('home'))
    
    from .plant_data import plants

    @app.route('/plants')
    def plant_list():
        return render_template('plants/list.html', plants=plants)

    @app.route('/plants/<plant_id>')
    def plant_detail(plant_id):
        plant = next((p for p in plants if p["id"] == plant_id), None)
        if not plant:
            return "Plant not found!", 404
        return render_template('plants/detail.html', plant=plant)

    return app



# Cho phép chạy trực tiếp bằng python flaskr/__init__.py
# if __name__ == '__main__':
#     app = create_app()
#     app.run(debug=True)

if __name__ == '__main__':
    app = create_app()
    from .chat import socketio
    socketio.run(app, debug=True)