"""What channels do the source files hold, and is a nuclear stain among them?

The ingest extracted only the mitochondrial channel. Per-cell analysis needs
nuclei, and inferring them from the perinuclear ring fails wherever that ring is
open, so a real nuclear channel would be preferable. This reports what the key
records about channel counts and dyes, without touching any column that could
carry condition.

Reads only sizeC, channel_name, channel_used and dye from the key. Prints counts
only, never a filename or a path.

    python3 scripts/check_channels.py
"""
import collections, csv, os

# the key sits beside the repo (mitofi/pipeline, mitofi/_BLIND_KEY...), never in it
KEY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), '_BLIND_KEY_do_not_open_until_rating_done',
    'tally_sheet_master.csv')
SAFE = ['sizeC', 'channel_name', 'channel_used', 'dye']
NUCLEAR = ('dapi', 'hoechst', '405', 'nucblue', 'draq')


def main():
    rows = [{k: r[k] for k in SAFE} for r in csv.DictReader(open(KEY)) if r['status'] == 'ok']
    print(f"images with status ok: {len(rows)}\n")
    for col in SAFE:
        c = collections.Counter(r[col].strip()[:32] or '(empty)' for r in rows)
        print(f"{col}:")
        for k, v in c.most_common(8):
            print(f"    {k:<34} {v}")
    hits = sum(1 for r in rows
               if any(n in (r['channel_name'] + ' ' + r['dye']).lower() for n in NUCLEAR))
    print(f"\nrows naming a nuclear stain in channel_name or dye: {hits}")
    if not hits:
        print("The key records only the extracted mitochondrial channel, so this cannot")
        print("confirm or exclude a nuclear channel. Files hold 3 or 4 channels; reading")
        print("the OME metadata of a few raw .oir directly is the way to settle it.")


if __name__ == '__main__':
    main()
