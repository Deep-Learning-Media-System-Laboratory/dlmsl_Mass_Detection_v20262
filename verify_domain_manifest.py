import csv
from collections import Counter
from pathlib import Path

MANIFEST = "domain_manifest.csv"


with open(
    MANIFEST,
    newline="",
    encoding="utf-8",
) as f:
    rows = list(csv.DictReader(f))


paths = [str(Path(r["image_path"]).resolve()) for r in rows]


print(
    "Rows:",
    len(rows),
)


duplicates = len(paths) - len(set(paths))


missing = [p for p in paths if not Path(p).exists()]


counts = Counter(row["domain"] for row in rows)


print(
    "Duplicate paths:",
    duplicates,
)

print(
    "Missing files:",
    len(missing),
)

print()
print("Domains:")

for domain, count in sorted(counts.items()):
    print(f"{domain:<20}{count:>10}")


assert duplicates == 0
assert len(missing) == 0
assert len(counts) >= 2


print()
print("PASS: manifest verified.")
