import os
import yaml
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent.parent

def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

class Config:
    def __init__(self):
        self.config_dir = BASE_DIR / "config"
        self.system = load_yaml(self.config_dir / "system.yaml")
        self.cameras = load_yaml(self.config_dir / "cameras.yaml").get("cameras", [])
        self.thresholds = load_yaml(self.config_dir / "thresholds.yaml")
        
        # Directories
        self.storage_dir = BASE_DIR / self.system.get("storage", {}).get("base_dir", "storage")
        self.snapshots_dir = BASE_DIR / self.system.get("storage", {}).get("snapshots_dir", "storage/snapshots")
        self.clips_dir = BASE_DIR / self.system.get("storage", {}).get("clips_dir", "storage/clips")
        self.face_db_dir = BASE_DIR / self.system.get("storage", {}).get("face_db_dir", "storage/face_db")
        self.models_dir = BASE_DIR / self.system.get("storage", {}).get("models_dir", "models")
        self.data_dir = BASE_DIR / self.system.get("storage", {}).get("data_dir", "data")
        
        self.database_url = os.getenv(
            "DATABASE_URL",
            self.system.get("database", {}).get("url", f"sqlite:///{self.storage_dir}/cctv_safety.db")
        )
        
        # Ensure directories exist
        for d in [self.storage_dir, self.snapshots_dir, self.clips_dir, self.face_db_dir, self.models_dir, self.data_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def reload(self):
        self.__init__()

config = Config()
