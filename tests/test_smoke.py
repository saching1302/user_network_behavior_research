from pathlib import Path
import pandas as pd

def test_sample_dataset_exists():
    p = Path("data/sample_user_network_behavior.csv")
    assert p.exists()
    df = pd.read_csv(p)
    assert len(df) > 0
    assert "user_id" in df.columns
