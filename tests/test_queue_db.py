from pathlib import Path

def get_db_base(tr):
    return Path(tr.get("queue_db") or "data/training.db").parent


# Test 1: Missing key
assert get_db_base({}) == Path("data")

# Test 2: None value
assert get_db_base({"queue_db": None}) == Path("data")

# Test 3: Empty string
assert get_db_base({"queue_db": ""}) == Path("data")

# Test 4: Valid custom path
assert get_db_base({"queue_db": "custom/my_training.db"}) == Path("custom")

# Test 5: Valid filename in current directory
assert get_db_base({"queue_db": "training.db"}) == Path(".")

print("All 5 queue_db tests passed!")