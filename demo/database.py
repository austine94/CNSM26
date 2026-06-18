import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from config import DB_PATH, POLICY_TXT

SCHEMA = """
CREATE TABLE IF NOT EXISTS viewers (
  viewer_id INTEGER PRIMARY KEY,
  longitude REAL NOT NULL,
  latitude REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS compute_nodes (
  node_id INTEGER PRIMARY KEY,
  longitude REAL NOT NULL,
  latitude REAL NOT NULL,
  capacity INTEGER NOT NULL,
  provision_financial_cost REAL NOT NULL,
  provision_energy_cost REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS viewer_node_costs (
  viewer_id INTEGER NOT NULL,
  node_id INTEGER NOT NULL,
  financial_cost REAL NOT NULL,
  energy_cost REAL NOT NULL,
  latency REAL NOT NULL,
  PRIMARY KEY (viewer_id, node_id)
);
CREATE TABLE IF NOT EXISTS ingestion_summary (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  last_viewer_ingest TEXT,
  last_compute_ingest TEXT,
  viewers_processed INTEGER NOT NULL DEFAULT 0,
  nodes_processed INTEGER NOT NULL DEFAULT 0,
  imputed_values INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS policies (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  priority TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS optimal_solutions (
  solution_id INTEGER PRIMARY KEY AUTOINCREMENT,
  solved_at TEXT NOT NULL,
  priority TEXT NOT NULL,
  alpha REAL NOT NULL,
  beta REAL NOT NULL,
  objective_value REAL,
  status TEXT NOT NULL,
  financial_cost REAL,
  energy_cost REAL,
  avg_latency REAL,
  open_nodes TEXT,
  assignments TEXT
);
"""

def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def connect(db_path: Path = DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def initialise_database(db_path: Path = DB_PATH):
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.execute("INSERT OR IGNORE INTO ingestion_summary (id) VALUES (1)")
        default_priority = POLICY_TXT.read_text().strip() if POLICY_TXT.exists() else "minimise financial cost"
        conn.execute(
            "INSERT OR IGNORE INTO policies (id, priority, updated_at) VALUES (1, ?, ?)",
            (default_priority, now_iso()),
        )
        conn.commit()

def _float_or_none(value):
    if value is None or value == "":
        return None
    return float(value)

def _mean_impute(records, fields):
    imputed = 0
    for field in fields:
        observed = [r[field] for r in records if r[field] is not None]
        fill = mean(observed) if observed else 0.0
        if field == "capacity":
            fill = max(1, int(round(fill)))
        for record in records:
            if record[field] is None:
                record[field] = fill
                imputed += 1
    return imputed

def _normalise(records, fields):
    for field in fields:
        values = [float(r[field]) for r in records]
        lo, hi = min(values), max(values)
        for record in records:
            record[field] = 0.0 if hi == lo else (float(record[field]) - lo) / (hi - lo)

def ingest_csvs(viewers_csv: Path, compute_nodes_csv: Path, db_path: Path = DB_PATH):
    initialise_database(db_path)
    viewers = []
    with open(viewers_csv, newline="") as f:
        for row in csv.DictReader(f):
            viewers.append({"viewer_id": int(row["viewer_id"]), "longitude": float(row["longitude"]), "latitude": float(row["latitude"])})

    nodes = []
    cost_rows = []
    with open(compute_nodes_csv, newline="") as f:
        for row in csv.DictReader(f):
            node_id = int(row["node_id"])
            nodes.append({
                "node_id": node_id,
                "longitude": float(row["longitude"]),
                "latitude": float(row["latitude"]),
                "capacity": _float_or_none(row.get("capacity")),
                "provision_financial_cost": _float_or_none(row.get("provision_financial_cost")),
                "provision_energy_cost": _float_or_none(row.get("provision_energy_cost")),
            })
            for viewer in viewers:
                vid = viewer["viewer_id"]
                cost_rows.append({
                    "viewer_id": vid,
                    "node_id": node_id,
                    "financial_cost": _float_or_none(row.get(f"viewer_{vid}_financial_cost")),
                    "energy_cost": _float_or_none(row.get(f"viewer_{vid}_energy_cost")),
                    "latency": _float_or_none(row.get(f"viewer_{vid}_latency")),
                })
    imputed = _mean_impute(nodes, ["capacity", "provision_financial_cost", "provision_energy_cost"])
    imputed += _mean_impute(cost_rows, ["financial_cost", "energy_cost", "latency"])
    for node in nodes:
        node["capacity"] = max(1, int(round(node["capacity"])))
    _normalise(nodes, ["provision_financial_cost", "provision_energy_cost"])
    _normalise(cost_rows, ["financial_cost", "energy_cost", "latency"])

    with connect(db_path) as conn:
        conn.execute("DELETE FROM viewers")
        conn.execute("DELETE FROM compute_nodes")
        conn.execute("DELETE FROM viewer_node_costs")
        conn.executemany("INSERT INTO viewers VALUES (:viewer_id, :longitude, :latitude)", viewers)
        conn.executemany("INSERT INTO compute_nodes VALUES (:node_id, :longitude, :latitude, :capacity, :provision_financial_cost, :provision_energy_cost)", nodes)
        conn.executemany("INSERT INTO viewer_node_costs VALUES (:viewer_id, :node_id, :financial_cost, :energy_cost, :latency)", cost_rows)
        conn.execute("UPDATE ingestion_summary SET last_viewer_ingest=?, last_compute_ingest=?, viewers_processed=?, nodes_processed=?, imputed_values=? WHERE id=1", (now_iso(), now_iso(), len(viewers), len(nodes), imputed))
        conn.commit()
    return {"viewers": len(viewers), "nodes": len(nodes), "imputed_values": imputed}

def set_policy(priority: str, db_path: Path = DB_PATH):
    POLICY_TXT.write_text(priority.strip() + "\n")
    initialise_database(db_path)
    with connect(db_path) as conn:
        conn.execute("INSERT INTO policies (id, priority, updated_at) VALUES (1, ?, ?) ON CONFLICT(id) DO UPDATE SET priority=excluded.priority, updated_at=excluded.updated_at", (priority.strip(), now_iso()))
        conn.commit()
