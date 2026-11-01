"""Inventory every series in every .lif under testing_data/test lifs, before scoring.

Metadata from Bio-Formats (OME-XML dumped by OmeDump.java, which reports laser
lines per channel); pixel layout and true bit depth from readlif. The two list
series in the same order (checked: names identical up to trailing spaces).

Channel rule, fixed before scoring: the mitochondrial channel is the one channel
excited between 550 and 600 nm (552, 561, 594 lines). None or several: the
series is listed as "channel not identifiable by laser line".

    .venv/bin/python validation/all_lif_2026-10/inventory.py XML_DIR
"""
import csv
import glob
import os
import sys
import xml.etree.ElementTree as ET

from readlif.reader import LifFile

HERE = os.path.dirname(os.path.abspath(__file__))
LIFS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                    'testing_data', 'test lifs')
LO, HI = 550.0, 600.0


def tag(e):
    return e.tag.rsplit('}', 1)[-1]


def main(xml_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(LIFS, '**', '*.lif'), recursive=True)):
        key = os.path.basename(p)[:-4].replace(' ', '_').replace('+', '_')
        imgs = [e for e in ET.parse(os.path.join(xml_dir, f'xml_{key}.ome.xml')).getroot()
                if tag(e) == 'Image']
        rl = LifFile(p).image_list
        assert len(imgs) == len(rl), p
        for i, (im, r) in enumerate(zip(imgs, rl)):
            assert im.get('Name').strip() == r['name'].strip(), (p, i)
            pix = next(e for e in im if tag(e) == 'Pixels')
            chs = [e for e in pix if tag(e) == 'Channel']
            ex = [c.get('ExcitationWavelength') for c in chs]
            hits = [k for k, w in enumerate(ex) if w and LO <= float(w) <= HI]
            sx = r['scale'][0]
            rows.append(dict(
                lif=os.path.relpath(p, LIFS), series=i, name=r['name'].strip(),
                x=r['dims'].x, y=r['dims'].y, z=r['dims'].z, t=r['dims'].t, m=r['dims'].m,
                c=r['channels'], bits=max(r['bit_depth']),
                px_um=round(1 / sx, 5) if sx else '',
                channel_names='|'.join(c.get('Name') or '' for c in chs),
                excitation_nm='|'.join(w or '' for w in ex),
                mito_channel=hits[0] if len(hits) == 1 else '',
                channel_rule='laser line' if len(hits) == 1 else
                ('channel not identifiable by laser line: '
                 + ('none' if not hits else 'several') + ' at 550-600 nm')))
    with open(os.path.join(HERE, 'inventory.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(len(rows), 'series')


if __name__ == '__main__':
    main(sys.argv[1])
