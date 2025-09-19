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