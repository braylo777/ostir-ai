CREATE TABLE IF NOT EXISTS design_partners (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT UNIQUE NOT NULL,
  node_count TEXT, deployment TEXT, note TEXT,
  ip TEXT, user_agent TEXT, created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_dp_created ON design_partners(created_at);
