import json
import os
from .snapshot_global_variables import snapshot_settings

class SnapshotRecord():
    def __init__(self, dir, rank):
        self.dir = dir
        self.rank = rank
        self.record_filename = os.path.join(self.dir, f"{self.rank}.json")


    def write(self, record):
        with open(self.record_filename, 'w') as f:
            json.dump(record, f)


    def read(self):
        if not os.path.exists(self.record_filename):
            return None

        with open(self.record_filename, 'r') as f:
            record = json.load(f)
        return record


    def construct_record(self, filename=None):
        record = {}
        record["rank"] = self.rank
        record["iteration"] = snapshot_settings.get_global_step()
        record["epoch"] = snapshot_settings.get_global_epoch()
        record["file"] = filename
        return record