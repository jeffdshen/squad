"""Write submission to file in csv format

Author:
    Chris Chute (chute@stanford.edu)
"""

import csv


def write_submission(sub_path, sub_dict):
    with open(sub_path, "w", newline="", encoding="utf-8") as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=",")
        csv_writer.writerow(["Id", "Predicted"])
        for uuid in sorted(sub_dict):
            csv_writer.writerow([uuid, sub_dict[uuid]])
