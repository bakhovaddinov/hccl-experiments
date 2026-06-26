import re
from collections import defaultdict

timeline = defaultdict(list)
current_time = None
current_npu = None

with open("npu_mem.log", "r") as f:
    for line in f:
        t_match = re.search(r'---\s+([\d\.]+)\s+---', line)
        if t_match:
            current_time = float(t_match.group(1))
            continue

        # Look for NPU ID definition
        npu_match = re.search(r'\|\s+(\d+)\s+910B4', line)
        if npu_match:
            current_npu = int(npu_match.group(1))
            continue

        # Look for memory, but only if we know which NPU we're in
        if current_npu is not None and current_time:
            # We look for the memory number in a line that contains a Bus-ID 
            # (which signifies it's the second line of the NPU block)
            mem_match = re.search(r'\|\s+0\s+\|\s+[\da-fA-F:]+\s+\|\s+\d+\s+\d+\s+/\s+\d+\s+(\d+)\s+/', line)
            if mem_match:
                used_mb = int(mem_match.group(1))
                timeline[current_npu].append((current_time, used_mb))
                # Do NOT reset current_npu here; wait for the next header