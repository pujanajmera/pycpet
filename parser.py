lines = []
with open("importtime.log") as f:
    for line in f:
        line = line.strip()
        if line.startswith("import time:"):
            # e.g. "import time:       161 |        161 |   _io"
            parts = line.split("|")
            if len(parts) == 3:
                self_us = parts[0].replace("import time:", "").strip()
                cumulative_us = parts[1].strip()
                modname = parts[2].strip()
                try:
                    self_us = int(self_us)
                    cumulative_us = int(cumulative_us)
                except ValueError:
                    continue
                lines.append((self_us, cumulative_us, modname))

# Sort descending by self import time
lines_sorted = sorted(lines, key=lambda x: x[0], reverse=True)

for i, (self_us, cumulative_us, modname) in enumerate(lines_sorted[:50]):
    print(f"{i+1:2d}. {self_us:8d} us self  | {cumulative_us:8d} us cum  | {modname}")
