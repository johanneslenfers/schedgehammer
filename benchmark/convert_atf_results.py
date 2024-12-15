# Only needed since this is in the same repo as schedgehammer.
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
##############################################################

from datetime import datetime, timedelta
from pathlib import Path
import json
import csv

for folder in os.listdir('results/atf'):
    Path(f'results/atf/{folder}/csv').mkdir(exist_ok=True)
    for i in range(50):
        with open(f'results/atf/{folder}/json/{i}.json', 'r') as json_file:
            with open(f'results/atf/{folder}/csv/ATF-{i}.csv', 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                # Header
                writer.writerow(["num_evaluation", "score", "timestamp"])
                # Body
                obj = json.load(json_file)
                for h in obj['history']:
                    t = datetime.strptime(h['timedelta_since_tuning_start'], '%H:%M:%S.%f')
                    delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
                    writer.writerow([h['evaluations'], h['cost'], delta.total_seconds()])