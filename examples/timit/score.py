import argparse
import json

import editdistance
import preprocess


def remap(data):
    _, m48_39 = preprocess.load_phone_map()
    for d in data:
        d["prediction"] = [m48_39[p] for p in d["prediction"]]
        d["label"] = [m48_39[p] for p in d["label"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CER on Timit with reduced phoneme set."
    )

    parser.add_argument("data_json", help="JSON with the transcripts.")
    args = parser.parse_args()

    with open(args.data_json, "r") as fid:
        data = [json.loads(l) for l in fid]

    remap(data)
    dist = sum(editdistance.eval(d["label"], d["prediction"]) for d in data)
    total = sum(len(d["label"]) for d in data)
    print(f"CER {dist / total:.3f}")
