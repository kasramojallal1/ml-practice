import pytest

@pytest.fixture
def sample_data():
    return {"name": "Alice", "age": 25}

def test_data(sample_data):
    assert sample_data["name"] == "Alice"
    assert sample_data["age"] == 25
