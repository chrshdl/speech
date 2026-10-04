import editdistance


def compute_cer(results):
    """
    Arguments:
        results (list): list of ground truth and
            predicted sequence pairs.

    Returns the CER for the full set.
    """
    dist = sum(editdistance.eval(label, pred) for label, pred in results)
    total = sum(len(label) for label, _ in results)
    return dist / total


def compute_wer(results):
    """
    Arguments:
        results (list): list of ground truth and predicted character
            sequence pairs, with words separated by spaces.

    Returns the WER for the full set.
    """
    words = [("".join(label).split(), "".join(pred).split()) for label, pred in results]
    return compute_cer(words)
