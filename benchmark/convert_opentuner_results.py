# Only needed since this is in the same repo as schedgehammer.
import csv
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
##############################################################

import sqlite3

db = sqlite3.connect('opentuner.db/DESKTOP-JUSTUS.db')
cur = db.execute("SELECT name, id, start_date FROM tuning_run")
last_name = None
i = 0
for x in cur:
    (name, id, start_date) = x
    start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S.%f')
    if last_name != name:
        last_name = name
        i = 0
    cur2 = db.execute("SELECT collection_date, time FROM result WHERE tuning_run_id = ? ORDER BY id", [id])
    filename = f"results/opentuner/{name}/OpenTuner-{i}.csv"
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        # Header
        writer.writerow(["num_evaluation", "score", "timestamp"])
        j = 0
        for record in cur2:
            seconds = (datetime.strptime(record[0], '%Y-%m-%d %H:%M:%S.%f') - start_date).total_seconds()
            writer.writerow([j, record[1], seconds])
            j += 1
    i += 1
