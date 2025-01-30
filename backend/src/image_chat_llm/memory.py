class Memory:
    def __init__(self):
        self.memory_store = {}

    def set(self, key: str, value: str):
        self.memory_store[key] = value

    def get(self, key: str):
        return self.memory_store.get(key, None)