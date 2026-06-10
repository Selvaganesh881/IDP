import pytest
from pathlib import Path

@pytest.fixture()
def config_yaml(tem_path: Path = Path("./tests")) -> Path:
    p = tem_path / "config_test.yaml"
    p.write_text(
        "model_name: roberta-base\n"
        "model_path: ./store_api/roberta-pii\n"
        "model_threshold: 0.5\n",
        encoding="utf-8"
    )
    
    return p


    
    
