import re
from collections import defaultdict

timeline = defaultdict(list)
current_time = None

with open("npu_mem.log", "r") as f:
    for line in f:
        t_match = re.search(r'---\s+([\d\.]+)\s+---', line)
        if t_match:
            current_time = float(t_match.group(1))
            continue

        mem_match = re.search(r'\|\s+(\d+)\s+\d+\s+\|\s+\S+\s+\|\s+\d+\s+(\d+)\s+/', line)
        if mem_match and current_time:
            npu_id = int(mem_match.group(1))
            used_mb = int(mem_match.group(2))
            timeline[npu_id].append((current_time, used_mb))

print("NPU 0:", timeline[0][:5])