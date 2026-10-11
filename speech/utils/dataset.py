import json


def read_data_json(data_json):
    """
    Reads a dataset json file, one example per line, or a list of them,
    which are concatenated.
    """
    paths = [data_json] if isinstance(data_json, str) else data_json
    data = []
    for path in paths:
        with open(path) as fid:
            data.extend(json.loads(l) for l in fid)
    return data
