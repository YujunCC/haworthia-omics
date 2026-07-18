import os
import sqlite3
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

DB_PATH = str(Path(os.getenv("HAWORTHIA_DB_PATH", "haworthia_omics.db")).expanduser())
IMG_DIR = str(Path(os.getenv("HAWORTHIA_IMAGE_DIR", "local_images")).expanduser())


def _remove_legacy_hybrid_column(conn):
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(taxonomy)").fetchall()
    }
    if "is_hybrid" not in columns:
        return None

    backup_dir = Path("backups") / "migrations"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = backup_dir / f"haworthia_omics_before_remove_hybrid_{timestamp}.db"
    backup_conn = sqlite3.connect(backup_path)
    try:
        conn.backup(backup_conn)
    finally:
        backup_conn.close()

    conn.execute("ALTER TABLE taxonomy DROP COLUMN is_hybrid")
    conn.commit()
    return str(backup_path.resolve())


def init_db():
    Path(DB_PATH).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(IMG_DIR).expanduser().mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS taxonomy
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  species TEXT, variant TEXT)''')
    conn.commit()
    _remove_legacy_hybrid_column(conn)
    c.execute('''CREATE TABLE IF NOT EXISTS images
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  tax_id INTEGER, orig_path TEXT, seg_path TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS prototypes
                 (tax_id INTEGER PRIMARY KEY, feature_blob BLOB)''')
    c.execute('''CREATE TABLE IF NOT EXISTS prototype_clusters
                 (tax_id INTEGER,
                  cluster_index INTEGER,
                  feature_blob BLOB,
                  sample_count INTEGER,
                  PRIMARY KEY (tax_id, cluster_index))''')
    conn.commit()
    conn.close()

def build_taxonomy_affinity_matrix():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, species, variant FROM taxonomy")
    taxa = c.fetchall()
    conn.close()

    id_to_idx = {t[0]: i for i, t in enumerate(taxa)}
    n = len(taxa)
    affinity = torch.zeros((n, n))
    for i, t1 in enumerate(taxa):
        for j, t2 in enumerate(taxa):
            if t1[0] == t2[0]:
                affinity[i, j] = 1.0
            elif t1[1] == t2[1]:
                affinity[i, j] = 0.5
            else:
                affinity[i, j] = 0.0
    return affinity, id_to_idx

def update_prototypes_in_db(class_features, prototypes_per_taxon=1):
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    from sklearn.cluster import KMeans

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for tax_id, feats in class_features.items():
        stacked = torch.stack(feats)
        mean_feat = stacked.mean(dim=0)
        mean_feat = F.normalize(mean_feat, p=2, dim=0)
        blob = mean_feat.numpy().astype(np.float32).tobytes()
        c.execute("REPLACE INTO prototypes (tax_id, feature_blob) VALUES (?, ?)", (tax_id, blob))

        c.execute("DELETE FROM prototype_clusters WHERE tax_id = ?", (tax_id,))
        cluster_count = min(max(1, prototypes_per_taxon), len(stacked))
        if cluster_count == 1:
            centers = mean_feat.unsqueeze(0).numpy()
            counts = [len(stacked)]
        else:
            feature_array = stacked.numpy().astype(np.float32)
            fitted = KMeans(n_clusters=cluster_count, n_init=10, random_state=42).fit(feature_array)
            centers = fitted.cluster_centers_.astype(np.float32)
            centers /= np.linalg.norm(centers, axis=1, keepdims=True).clip(min=1e-8)
            counts = np.bincount(fitted.labels_, minlength=cluster_count).tolist()

        for cluster_index, (center, sample_count) in enumerate(zip(centers, counts)):
            c.execute(
                "INSERT INTO prototype_clusters "
                "(tax_id, cluster_index, feature_blob, sample_count) VALUES (?, ?, ?, ?)",
                (tax_id, cluster_index, center.astype(np.float32).tobytes(), int(sample_count))
            )
    conn.commit()
    conn.close()

def get_all_prototypes():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""SELECT taxonomy.id, taxonomy.species, taxonomy.variant, prototypes.feature_blob
                 FROM prototypes JOIN taxonomy ON prototypes.tax_id = taxonomy.id
                 ORDER BY taxonomy.id""")
    rows = c.fetchall()
    conn.close()
    return rows


def get_all_cluster_prototypes():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""SELECT taxonomy.id, taxonomy.species, taxonomy.variant,
                        prototype_clusters.cluster_index,
                        prototype_clusters.feature_blob,
                        prototype_clusters.sample_count
                 FROM prototype_clusters
                 JOIN taxonomy ON prototype_clusters.tax_id = taxonomy.id
                 ORDER BY taxonomy.id, prototype_clusters.cluster_index""")
    rows = c.fetchall()
    if not rows:
        c.execute("""SELECT taxonomy.id, taxonomy.species, taxonomy.variant,
                            0, prototypes.feature_blob, 0
                     FROM prototypes JOIN taxonomy ON prototypes.tax_id = taxonomy.id
                     ORDER BY taxonomy.id""")
        rows = c.fetchall()
    conn.close()
    return rows

# 追加至 database.py 尾部

def insert_taxonomy(species: str, variant: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM taxonomy WHERE species=? AND variant=?",
              (species, variant))
    existing = c.fetchone()
    if existing:
        tax_id = existing[0]
        is_new = False
    else:
        c.execute("INSERT INTO taxonomy (species, variant) VALUES (?, ?)",
                  (species, variant))
        tax_id = c.lastrowid
        is_new = True
    conn.commit()
    conn.close()
    return tax_id, is_new

def get_image_count_by_tax(tax_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM images WHERE tax_id=?", (tax_id,))
    count = c.fetchone()[0]
    conn.close()
    return count

def insert_image_record(tax_id: int, orig_path: str, seg_path: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO images (tax_id, orig_path, seg_path) VALUES (?, ?, ?)",
              (tax_id, orig_path, seg_path))
    conn.commit()
    conn.close()

def get_taxonomy_records():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, species, variant FROM taxonomy ORDER BY id DESC")
    records = c.fetchall()
    conn.close()
    return records

def delete_taxonomy_cascade(tax_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT orig_path, seg_path FROM images WHERE tax_id = ?", (tax_id,))
    paths = c.fetchall()
    for op, sp in paths:
        if os.path.exists(op): os.remove(op)
        if os.path.exists(sp): os.remove(sp)
    c.execute("DELETE FROM images WHERE tax_id = ?", (tax_id,))
    c.execute("DELETE FROM prototypes WHERE tax_id = ?", (tax_id,))
    c.execute("DELETE FROM prototype_clusters WHERE tax_id = ?", (tax_id,))
    c.execute("DELETE FROM taxonomy WHERE id = ?", (tax_id,))
    conn.commit()
    conn.close()


def get_all_image_records():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, orig_path, seg_path FROM images")
    records = c.fetchall()
    conn.close()
    return records


def get_image_records_by_tax(tax_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, tax_id, orig_path, seg_path FROM images WHERE tax_id = ? ORDER BY id",
        (tax_id,),
    )
    records = c.fetchall()
    conn.close()
    return records


def get_image_record(image_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, tax_id, orig_path, seg_path FROM images WHERE id = ?",
        (image_id,),
    )
    record = c.fetchone()
    conn.close()
    return record


def delete_image_record(image_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM images WHERE id = ?", (image_id,))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return deleted > 0


def get_database_overview():
    """Return lightweight database health and coverage statistics for the UI."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    taxonomy_count = c.execute("SELECT COUNT(*) FROM taxonomy").fetchone()[0]
    species_count = c.execute(
        "SELECT COUNT(DISTINCT species) FROM taxonomy"
    ).fetchone()[0]
    image_count = c.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    prototype_count = c.execute("SELECT COUNT(*) FROM prototypes").fetchone()[0]
    cluster_count = c.execute(
        "SELECT COUNT(*) FROM prototype_clusters"
    ).fetchone()[0]
    cluster_taxon_count = c.execute(
        "SELECT COUNT(DISTINCT tax_id) FROM prototype_clusters"
    ).fetchone()[0]
    distribution = c.execute(
        """
        SELECT taxonomy.id, taxonomy.species, taxonomy.variant,
               COUNT(images.id) AS image_count
        FROM taxonomy
        LEFT JOIN images ON images.tax_id = taxonomy.id
        GROUP BY taxonomy.id
        ORDER BY image_count DESC, taxonomy.id
        """
    ).fetchall()
    image_paths = c.execute("SELECT orig_path, seg_path FROM images").fetchall()
    conn.close()

    counts = [row[3] for row in distribution]
    missing_files = sum(
        not (os.path.exists(orig_path) and os.path.exists(seg_path))
        for orig_path, seg_path in image_paths
    )
    return {
        "taxonomy_count": taxonomy_count,
        "species_count": species_count,
        "image_count": image_count,
        "prototype_count": prototype_count,
        "cluster_count": cluster_count,
        "cluster_taxon_count": cluster_taxon_count,
        "taxa_without_prototypes": max(taxonomy_count - prototype_count, 0),
        "taxa_without_images": sum(count == 0 for count in counts),
        "missing_files": int(missing_files),
        "image_min_per_taxon": min(counts) if counts else 0,
        "image_max_per_taxon": max(counts) if counts else 0,
        "image_mean_per_taxon": (
            round(sum(counts) / len(counts), 2) if counts else 0.0
        ),
        "distribution": [
            {
                "id": row[0],
                "species": row[1],
                "variant": row[2],
                "image_count": row[3],
            }
            for row in distribution
        ],
    }
