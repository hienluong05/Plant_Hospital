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
from flask import session

def get_class_names_from_folder(dataset_dir):
    class_names = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    class_names.sort()
    return class_names

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
            print("Database chưa tồn tại, tự động khởi tạo từ schema.sql...")
            db.init_db()
    
    # Đăng ký blueprint auth
    from . import auth
    app.register_blueprint(auth.bp)

    # Helper function để get database
    def get_db():
        return db.get_db()
    
    chat.init_socketio(app) 

    # Route cho trang chủ (không cần đăng nhập)
    @app.route('/')
    def home():
        return render_template('home.html')

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
            
            flash('Câu hỏi đã được gửi thành công! Chúng tôi sẽ phản hồi trong vòng 24-48 giờ.', 'success')
            return redirect(url_for('expert_consultation'))
            
        except Exception as e:
            flash('Có lỗi xảy ra. Vui lòng thử lại sau.', 'error')
            return redirect(url_for('expert_consultation'))

    # Route cho book visit
    @app.route('/book-visit', methods=['POST'])
    def book_visit():
        try:
            name = request.form['visit_name']
            email = request.form['visit_email']
            phone = request.form['visit_phone']
            address = request.form['visit_address']
            visit_date = request.form['visit_date']
            visit_time = request.form['visit_time']
            garden_size = request.form['garden_size']
            
            # Lưu vào database
            database = get_db()
            database.execute(
                'INSERT INTO visits (name, email, phone, address, visit_date, visit_time, garden_size, created_at, status)'
                ' VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (name, email, phone, address, visit_date, visit_time, garden_size, 
                 datetime.now(), 'pending')
            )
            database.commit()
            
            flash('Đặt lịch thành công! Chúng tôi sẽ liên hệ xác nhận trong thời gian sớm nhất.', 'success')
            return redirect(url_for('expert_consultation'))
            
        except Exception as e:
            flash('Có lỗi xảy ra. Vui lòng thử lại sau.', 'error')
            return redirect(url_for('expert_consultation'))

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
        return render_template('chat.html')

    return app



# Cho phép chạy trực tiếp bằng python flaskr/__init__.py
# if __name__ == '__main__':
#     app = create_app()
#     app.run(debug=True)

if __name__ == '__main__':
    app = create_app()
    from .chat import socketio
    socketio.run(app, debug=True)