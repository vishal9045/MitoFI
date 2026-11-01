"""The label the shipped model is fit to, from the three blind raters.

blinded_call is the call at least two of the three made, on images where all
three made a morphology call. Where any rater said uncertain it is `uncertain`,
and where all three disagreed it is `split`. Neither is trained on. Dropping
every image any rater could not read, rather than letting two readers outvote
one, is the stricter of the two sets scripts/rater_agreement.py compares.

    python3 scripts/consensus_labels.py
"""
import os
from collections import Counter
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = {'fragmented', 'normal', 'elongated'}


def consensus(calls):
    if not set(calls) <= MAIN:
        return 'uncertain'
    call, n = Counter(calls).most_common(1)[0]
    return call if n >= 2 else 'split'


def main():
    d = None
    for i, name in enumerate(('labels_blind.csv', 'labels_rater2.csv', 'labels_rater3.csv'), 1):
        r = pd.read_csv(os.path.join(ROOT, 'data', name))[['image_id', 'blinded_call']]
        r = r.rename(columns={'blinded_call': f'r{i}'})
        d = r if d is None else d.merge(r, on='image_id', how='outer', validate='1:1')
    assert len(d) == 500 and d.notna().all().all(), 'every rater must rate every image'
    d.insert(1, 'blinded_call', [consensus(c) for c in d[['r1', 'r2', 'r3']].values])
    d.to_csv(os.path.join(ROOT, 'data', 'labels_consensus.csv'), index=False)
    print(d.blinded_call.value_counts().to_string())
    print('-> data/labels_consensus.csv')


if __name__ == '__main__':
    assert consensus(['normal', 'normal', 'fragmented']) == 'normal'
    assert consensus(['normal', 'elongated', 'fragmented']) == 'split'
    assert consensus(['normal', 'normal', 'uncertain']) == 'uncertain'
    main()
