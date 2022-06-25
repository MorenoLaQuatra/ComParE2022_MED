from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
import argparse
import pandas as pd

parser = argparse.ArgumentParser()


parser.add_argument(
    "--ground_truth_csv_path", help="Path pointing to the ground-truth csv.", required=True
)

parser.add_argument(
    "--prediction_csv_path", help="Path pointing to the metadata csv.", required=True
)

args = parser.parse_args()

gt = pd.read_csv(args.ground_truth_csv_path, sep="\t")
pred = pd.read_csv(args.prediction_csv_path, sep="\t")
metric = DiarizationErrorRate()

unique_ids = pred["filename"].tolist()
unique_ids.extend(gt["filename"].tolist())

print(unique_ids)
unique_ids = list(set(unique_ids))

error_rates = []
for id_file in unique_ids: # for each file in the dataset
    print(id_file)
    try:
        manual_reference = Annotation()
        for index in gt.index:
            if gt.loc[index,'filename'] == id_file:
                onset = gt.loc[index,"onset"]
                offset = gt.loc[index,"offset"]
                speaker = gt.loc[index,"event_label"]
                manual_reference[Segment(onset, offset)] = speaker

        # for row in gt[gt['filename'] == id_file]:
        #     print(row)
        #     onset = row["onset"]
        #     offset = row["offset"]
        #     speaker = row["event_label"]
        #     manual_reference[Segment(onset, offset)] = speaker

        automatic_hypothesis = Annotation()
        for index in pred.index:
            if pred.loc[index,'filename'] == id_file:
                onset = pred.loc[index,"onset"]
                offset = pred.loc[index,"offset"]
                speaker = pred.loc[index,"event_label"]
                automatic_hypothesis[Segment(onset, offset)] = speaker

        error_rate = metric(manual_reference, automatic_hypothesis)
        error_rates.append(error_rate)
    except Exception as e:
        print(e)

global_value = abs(metric)
mean, (lower, upper) = metric.confidence_interval()
print ("DER: ", global_value)
print ("Avg error rate: ", sum(error_rates)/len(error_rates))
print (f"mean {mean} - CI {lower},{upper}")