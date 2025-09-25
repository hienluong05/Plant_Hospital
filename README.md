# Plant Hospital 🌱🩺

**Plant Hospital** is an integrated web-based platform for AI-powered plant disease diagnosis, expert consultation, pharmacy management, blogging, and more—designed for farmers, agronomists, and plant enthusiasts.

---

## Key Features

- **AI Plant Disease Diagnosis**
  - Upload plant leaf images and receive instant disease predictions via a deep learning (ResNet50) model.
  - User-friendly interface and high accuracy.

- **Expert Consultation**
  - **Free Q&A:** Submit questions (with optional images) and get expert responses via email.
  - **Home Visit Booking:** Schedule on-site consultations with plant health experts.
  - **Live Chat:** Real-time chat with experts (websockets), with message history and image sharing.

- **Plant Pharmacy & E-Commerce**
  - Browse, search, and filter plant medicine and fertilizer products.
  - Add items to shopping cart, place orders, and track order statuses.
  - Admin dashboard for managing products and orders.

- **Blog & News**
  - Read and write blog posts related to agriculture and plant care.
  - Personal blog management for users.
  - Aggregated news feed from reputable agriculture sources.

- **Admin Dashboard**
  - Track statistics: total questions, consultations, orders, expert ratings, etc.
  - Manage all products, orders, chat sessions, and Q&A.
  - Top experts leaderboard.

- **Plant Encyclopedia**
  - Browse detailed information on a variety of crops: origin, growth cycle, nutrition, care tips, and common diseases with treatment advice.

---

## Technology Stack

- **Backend:** Python Flask, Flask-SocketIO, SQLite
- **Frontend:** HTML5, CSS3, Bootstrap, Jinja2 Templates, FontAwesome
- **AI Model:** PyTorch, torchvision, ResNet50
- **Realtime:** WebSockets (chat)
- **Authentication:** Flask-Login, hashed passwords
- **Payment (optional):** PayPal SDK (ready for integration)

---

## Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/hienluong05/Plant_Hospital.git
   cd Plant_Hospital/MyAI/App
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run database migrations**
   ```bash
   flask --app flaskr init-db
   ```

5. **Start the application**
   ```bash
   flask --app flaskr run
   # Or for realtime chat:
   python -m flaskr
   ```

6. **Access the app:**  
   Open [http://127.1.0.0:5000](http://127.1.0.0:5000) in your browser.

---

## Sample Accounts

- **Admin:**  
  - Email: (register as first user)
- **Expert:**  
  - Email: ms.victorialewis@gmail.com  
  - Password: 123456
  - [Add more experts in the database as needed]

---

## Project Structure

```
MyAI/App/
  flaskr/
    templates/      # Jinja2 HTML templates
    static/         # Static files (CSS, JS, images)
    __init__.py     # App factory, routes, AI inference
    db.py           # Database helpers
    auth.py         # Authentication logic
    admin.py        # Admin interface
    expert.py       # Expert interface
    blog.py         # Blog module
    chat.py         # WebSocket chat logic
    plant_data.py   # Plant encyclopedia data
    schema.sql      # Database schema
```

---

## Screenshots

> **Homepage, AI diagnosis, pharmacy UI, expert chat, and admin dashboard—see `/templates/` for UI details.**

---

## License

MIT License

---

**Contact:**  
- Email: ms.dangthihienluong@gmail.com  
- Facebook: [dangthihienluong](https://facebook.com/dangthihienluong/)  
- Twitter: [Hien_Luong_pre](https://x.com/Hien_Luong_pre)

---

**Plant Hospital – Modern plant healthcare, powered by AI and community expertise.**
