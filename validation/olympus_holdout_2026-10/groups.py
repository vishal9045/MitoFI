"""Comparisons PLAN.md fixes, as file selections; writes inputs.csv.

Every file the channel check kept is listed (group '' when it is in no
comparison: scored and counted, not tested). Selections use the source folder
and the file name, nothing else.

    .venv/bin/python validation/olympus_holdout_2026-10/groups.py <holdout dir>
"""
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CC = '19-09-2024 HUH-7 CCCP MitoGFP-RFP Different conditions/'
G = '27-12-2023/Mito GFP-RFP/'

# id, direction (frag: first above second; elong: first below second; none),
# confidence (firm / tentative / none), first (folder, name regex), second (folder, name regex)
COMPARISONS = [
    ('A', 'frag', 'firm', ('06-11-2023', r'CCCP 1 in 200'), ('06-11-2023', r'Mock 1 in 200')),
    ('B', 'frag', 'firm', (CC + 'EV + CCCP', r''), (CC + 'EV MOCK', r'')),
    ('C12', 'frag', 'firm', ('17-07-2023/MAVS 12 HPI', r'^Inf'), ('17-07-2023/MAVS 12 HPI', r'^Mock')),
    ('C24', 'frag', 'firm', ('17-07-2023/MAVS 24 HPI', r'^Infected'), ('17-07-2023/MAVS 24 HPI', r'^Mock')),
    ('C48', 'frag', 'firm', ('17-07-2023/MAVS 48 HPI', r'^Infected'), ('17-07-2023/MAVS 48 HPI', r'^Mock')),
    ('D12', 'frag', 'tentative', ('29-12-2022', r'12 hpi infected'), ('29-12-2022', r'12 hpi mock')),
    ('D24', 'frag', 'tentative', ('29-12-2022', r'24 hpi infected'), ('29-12-2022', r'24 hpi mock')),
    ('D48', 'frag', 'tentative', ('30-12-2022', r'48 hpi inf'), ('30-12-2022', r'48 hpi mock')),
    ('E', 'frag', 'firm', ('Mdivi experiment/Vehicle infected', r''), ('Mdivi experiment/Vehicle Mock', r'')),
    ('F_CoV2', 'frag', 'firm', ('27-11-2023/Huh-7 CoV-2', r''), ('27-11-2023/Huh-7 Mock', r'')),
    ('F_JEV', 'frag', 'firm', ('27-11-2023/Huh-7 JEV', r''), ('27-11-2023/Huh-7 Mock', r'')),
    ('F_DENV', 'elong', 'tentative', ('27-11-2023/Huh-7 DENV', r''), ('27-11-2023/Huh-7 Mock', r'')),
    ('G_CoV2', 'frag', 'firm', (G + 'CoV-2 24 hpi', r''), (G + 'Mock 24 hpi', r'')),
    ('G_JEV', 'frag', 'firm', (G + 'JEV 24 hpi', r''), (G + 'Mock 24 hpi', r'')),
    ('G_DENV', 'elong', 'tentative', (G + 'DENV 24 hpi', r''), (G + 'Mock 24 hpi', r'')),
    ('I', 'frag', 'firm', ('09-05-2024 Mito GFP RFP WT KO', r'MAVS KO Mock'), ('09-05-2024 Mito GFP RFP WT KO', r'WT Mock')),
    ('J_PUNO', 'none', 'none', ('10-07-2023/A549 KO EV', r''), ('10-07-2023/A549 WT EV', r'')),
    ('J_polyIC', 'none', 'none', ('10-07-2023/A549 KO POLY IC', r''), ('10-07-2023/A549 WT POLY IC', r'')),
    ('K_mock', 'frag', 'firm', ('22-01-2024', r'MAVS KO 24 hpi endo MAVS Mock F\d+ Zoom$'), ('22-01-2024', r'WT 24 hpi endo MAVS Mock')),
    ('K_CoV2', 'frag', 'firm', ('22-01-2024', r'MAVS KO\s+24 hpi endo MAVS CoV-2'), ('22-01-2024', r'WT 24 hpi endo MAVS CoV-2')),
    ('L_Dec', 'elong', 'firm', ('18-12-2023/huh-7 IFN treatment endo LC3', r'IFN treatment'), ('18-12-2023/huh-7 IFN treatment endo LC3', r'Vehicle treatment')),
    ('L_Apr', 'elong', 'firm', ('29-04-2024', r'IFN treatment'), ('29-04-2024/Vehicle control IFN', r'')),
    ('M', 'elong', 'firm', ('19-09-2022', r'polyIC mock .*2\.5x'), ('19-09-2022', r'PBS mock .*2\.5x')),
    ('M_4x', 'elong', 'secondary', ('19-09-2022', r'polyIC mock .*4x'), ('19-09-2022', r'PBS mock .*4x')),
    ('N_CoV2', 'elong', 'firm', ('29-01-2024/MAVS oe Mito GFP CoV-2', r'MAVS oe .*(?<!diff conc R)$'), ('29-01-2024/MAVS oe Mito GFP CoV-2', r'PUNO')),
    ('N_LC3', 'elong', 'firm', ('18-12-2023/HUH-7 MAVS oe ENDO LC3', r'MAVS oe'), ('18-12-2023/HUH-7 MAVS oe ENDO LC3', r'PUNO oe')),
    ('O', 'elong', 'firm', ('MAVS JEV MitoGFP-RFP/JEV FL-MAVS', r''), ('MAVS JEV MitoGFP-RFP/JEV EV', r'')),
    ('P', 'none', 'none', ('08-04-2024/MEF MAVS KO MFN1 oe', r'MFN1 oe'), ('08-04-2024/MEF MAVS KO MFN1 oe', r'EV oe')),
    ('Q', 'none', 'none', ('20-09-2022 Dixit|23-09-2022', r'(?i)a2a'), ('20-09-2022 Dixit|23-09-2022', r'(?i)mock')),
]


def pick(e, sel):
    folder, rx = sel
    f = e.folder.isin(folder.split('|'))          # exact folder names, '|' separates several
    return e[f & e.name.str.contains(rx, regex=True)]


def main(H):
    c = pd.read_csv(os.path.join(H, 'channel_calls.csv'))
    keep = c[c.channel != 'drop'].set_index('folder')
    e = pd.read_pickle(os.path.join(H, 'eligible.pkl'))
    e = e[e.folder.isin(keep.index)].copy()
    e['name'] = e.rel.map(lambda r: os.path.splitext(os.path.basename(r))[0])
    e['channel_ex'] = e.folder.map(lambda f: 500 if keep.loc[f, 'channel'].startswith('ch2 GFP') else 570)
    rows = []
    for cid, d, conf, a, b in COMPARISONS:
        for side, sel in (('first', a), ('second', b)):
            for _, r in pick(e, sel).iterrows():
                rows.append(dict(comparison=cid, side=side, direction=d, confidence=conf, rel=r.rel))
    g = pd.DataFrame(rows)
    assert not g.duplicated(['comparison', 'rel']).any()
    both = g.groupby(['comparison', 'rel']).side.nunique()
    assert (both == 1).all(), both[both > 1]
    out = e[['rel', 'folder', 'name', 'day', 'px', 'Z', 'C', 'channel_ex']].merge(g, on='rel', how='left')
    out.to_csv(os.path.join(HERE, 'inputs.csv'), index=False)
    n = g.groupby(['comparison', 'side']).size().unstack()
    print(n.loc[[c[0] for c in COMPARISONS]].to_string())
    print(len(e), 'files;', g.rel.nunique(), 'in comparisons')


if __name__ == '__main__':
    main(sys.argv[1])
