import pytest
from pathlib import Path
from pydantic import ValidationError

from settings import Settings, load_config_yaml, get_settings

def test_load_config(config_yaml: Path):
    data = load_config_yaml(config_yaml)
    assert isinstance(data, dict)
    assert data["model_threshold"] == 0.5
    
def test_model_threshold():
    with pytest.raises(ValidationError):
        Settings(model_threshold=1.1)
        
def test_config_cache(config_yaml: Path):
    get_settings.cache_clear()
    s1 = get_settings(str(config_yaml))
    s2 = get_settings(str(config_yaml))
    assert s1 is s2