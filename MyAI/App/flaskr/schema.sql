-- Drop existing tables if they exist
DROP TABLE IF EXISTS visits;
DROP TABLE IF EXISTS questions;
DROP TABLE IF EXISTS experts;
DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS post;

-- User table for authentication
CREATE TABLE user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  email TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Post table (keeping your existing structure)
CREATE TABLE post (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  author_id INTEGER NOT NULL,
  created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  FOREIGN KEY (author_id) REFERENCES user (id)
);

-- Questions table for expert consultation
CREATE TABLE questions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  email TEXT NOT NULL,
  phone TEXT,
  plant_type TEXT NOT NULL,
  question TEXT NOT NULL,
  images TEXT,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  status TEXT NOT NULL DEFAULT 'pending',
  response TEXT,
  expert_id INTEGER,
  responded_at TIMESTAMP,
  FOREIGN KEY (expert_id) REFERENCES experts (id)
);

-- Visits table for expert home consultation booking
CREATE TABLE visits (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  email TEXT NOT NULL,
  phone TEXT NOT NULL,
  address TEXT NOT NULL,
  visit_date DATE NOT NULL,
  visit_time TEXT NOT NULL,
  garden_size TEXT,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  status TEXT NOT NULL DEFAULT 'pending',
  expert_id INTEGER,
  confirmed_at TIMESTAMP,
  notes TEXT,
  FOREIGN KEY (expert_id) REFERENCES experts (id)
);

-- Experts table for managing consultants
CREATE TABLE experts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  email TEXT NOT NULL,
  password TEXT NOT NULL,  -- thêm dòng này!
  phone TEXT,
  specialties TEXT,
  bio TEXT,
  avatar TEXT,
  rating REAL DEFAULT 5.0,
  active BOOLEAN DEFAULT 1,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
-- Insert sample experts
INSERT INTO experts (name, email, password, phone) 
VALUES ('Dr. Victoria Lewis', 'ms.victorialewis@gmail.com', '123456', '0901234567');

-- Bảng lưu phiên chat (giữa 1 khách và 1 chuyên gia)
CREATE TABLE chat_session (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    expert_id INTEGER NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_message_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'active',
    FOREIGN KEY(user_id) REFERENCES user(id),
    FOREIGN KEY(expert_id) REFERENCES experts(id)
);

-- Bảng lưu từng tin nhắn
CREATE TABLE chat_message (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    sender_type TEXT NOT NULL, -- 'user' hoặc 'expert'
    sender_id INTEGER,         -- id của người gửi (user hoặc expert)
    content TEXT,
    image_path TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_read BOOLEAN DEFAULT 0,
    FOREIGN KEY(session_id) REFERENCES chat_session(id)
);

-- Table lưu đơn mua thuốc
CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER,
  total REAL,
  status TEXT DEFAULT 'pending', -- pending/paid/cancelled
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS order_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  order_id INTEGER,
  product_name TEXT,
  price REAL,
  quantity INTEGER
);

-- Table lưu giao dịch thanh toán
CREATE TABLE IF NOT EXISTS payments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER,
  service_type TEXT, -- 'pharmacy', 'consultation'
  service_id INTEGER, -- id của order hoặc booking
  amount REAL,
  status TEXT DEFAULT 'pending', -- pending/paid/failed
  method TEXT, -- paypal, vnpay...
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);