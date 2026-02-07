from pathlib import Path


def test_examples_txt_shape():
    path = Path("data/examples.txt")
    assert path.exists()
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 10
    for row in rows:
        parts = row.split("|")
        assert len(parts) == 4
        assert parts[0].strip()
        assert parts[1].strip()
        assert parts[2].strip()
        assert parts[3].strip()
