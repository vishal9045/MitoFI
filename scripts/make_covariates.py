"""Extract the non-revealing columns from the sealed key into data/covariates.csv.

The blind is protected by taking only these columns:
    random_id, session, px_um, sizeZ, z_step_um, z_range_um, px_mean
and by relabelling sessions as S01..Snn in a seeded shuffle, so neither the
label nor its ordering says anything about acquisition date or group.

Deliberately NOT taken: arm, original_file, blind_file, source_path, key_source,
flag_both_conditions -- every column that could carry condition.

Rerunning with the same seed reproduces identical session labels.
"""
import csv, os, random

# the key sits beside the repo (mitofi/pipeline, mitofi/_BLIND_KEY...), never in it
KEY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), '_BLIND_KEY_do_not_open_until_rating_done',
    'tally_sheet_master.csv')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = os.path.join(ROOT, 'data', 'labels_blind.csv')
OUT = os.path.join(ROOT, 'data', 'covariates.csv')
KEEP = ['random_id', 'session', 'px_um', 'sizeZ', 'z_step_um', 'z_range_um', 'px_mean']
SEED = 20260905


def norm(rid):
    return rid if rid.startswith('img_') else 'img_' + rid


def main():
    rows = [{k: r[k] for k in KEEP} for r in csv.DictReader(open(KEY)) if r['status'] == 'ok']
    rated = {r['image_id'] for r in csv.DictReader(open(LABELS))}

    sessions = sorted({r['session'] for r in rows})
    shuffled = sessions[:]
    random.Random(SEED).shuffle(shuffled)
    label = {s: f"S{i+1:02d}" for i, s in enumerate(shuffled)}

    out = [r for r in rows if norm(r['random_id']) in rated]
    with open(OUT, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['image_id', 'session', 'px_um', 'sizeZ', 'z_step_um', 'z_range_um', 'px_mean'])
        for r in out:
            w.writerow([norm(r['random_id']), label[r['session']], r['px_um'], r['sizeZ'],
                        r['z_step_um'], r['z_range_um'], r['px_mean']])

    print(f"key rows with status=ok : {len(rows)}")
    print(f"rated images            : {len(rated)}")
    print(f"matched and written     : {len(out)}")
    print(f"sessions                : {len(sessions)}")
    assert len(out) == len(rated), "some rated image is missing from the key"


if __name__ == '__main__':
    main()
