-- Sample e-commerce database WITHOUT declared FKs
-- Agent will discover these relationships

CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT
);

CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE suppliers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    contact_email VARCHAR(255)
);

-- FK to categories (not declared)
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    category_id INTEGER NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- FK to customers (not declared)
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(10,2)
);

-- FKs to orders and products (not declared)
CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL
);

-- Junction table - FKs to products and suppliers (not declared)
CREATE TABLE product_suppliers (
    id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL,
    supplier_id INTEGER NOT NULL,
    is_primary BOOLEAN DEFAULT FALSE
);

-- Sample data
INSERT INTO categories (name, description) VALUES
    ('Electronics', 'Electronic devices'),
    ('Clothing', 'Apparel'),
    ('Books', 'Books'),
    ('Home', 'Home goods');

INSERT INTO customers (email, name) VALUES
    ('alice@example.com', 'Alice Johnson'),
    ('bob@example.com', 'Bob Smith'),
    ('carol@example.com', 'Carol Williams'),
    ('david@example.com', 'David Brown'),
    ('eve@example.com', 'Eve Davis');

INSERT INTO suppliers (name, contact_email) VALUES
    ('TechSupply', 'sales@techsupply.com'),
    ('FashionDirect', 'orders@fashion.com'),
    ('BookDist', 'info@bookdist.com'),
    ('HomeEssentials', 'contact@home.com');

INSERT INTO products (name, category_id, price) VALUES
    ('Laptop', 1, 1299.99),
    ('Mouse', 1, 29.99),
    ('USB Hub', 1, 49.99),
    ('Jacket', 2, 89.99),
    ('T-Shirt', 2, 24.99),
    ('Shoes', 2, 119.99),
    ('Python Book', 3, 49.99),
    ('Data Science', 3, 59.99),
    ('Garden Tools', 4, 79.99),
    ('Plant Kit', 4, 34.99);

INSERT INTO orders (customer_id, total_amount) VALUES
    (1, 1329.98), (1, 89.99), (2, 154.98),
    (3, 59.99), (4, 244.98), (5, 79.99);

INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
    (1, 1, 1, 1299.99), (1, 2, 1, 29.99),
    (2, 4, 1, 89.99), (3, 5, 2, 24.99),
    (3, 7, 1, 49.99), (3, 8, 1, 59.99),
    (4, 8, 1, 59.99), (5, 6, 1, 119.99),
    (5, 9, 1, 79.99), (5, 10, 1, 34.99),
    (6, 9, 1, 79.99);

INSERT INTO product_suppliers (product_id, supplier_id, is_primary) VALUES
    (1, 1, TRUE), (2, 1, TRUE), (3, 1, TRUE),
    (4, 2, TRUE), (5, 2, TRUE), (6, 2, TRUE),
    (7, 3, TRUE), (8, 3, TRUE),
    (9, 4, TRUE), (10, 4, TRUE);

ANALYZE;