"""Stage 3 label audit: integrity, call distribution, tag counts, order-effect check.

Every number this prints appears in notes/pilot_record_draft.docx section 5.
Run it to reproduce them.

    python3 scripts/check_labels.py
"""
import collections, csv, math, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = os.path.join(ROOT, 'data', 'labels_blind.csv')
RATER = os.path.join(ROOT, 'data', 'blind_rater.html')
PNGDIR = os.path.join(ROOT, 'data', 'rating_png')

CALLS = ['fragmented', 'normal', 'elongated', 'uncertain']
TAGS = ('donut', 'clump')


def chi2(table):
    """Pearson chi-square and degrees of freedom for a contingency table."""
    rowsum = [sum(r) for r in table]
    colsum = [sum(c) for c in zip(*table)]
    total = sum(rowsum)
    x = 0.0
    for i, row in enumerate(table):
        for j, obs in enumerate(row):
            exp = rowsum[i] * colsum[j] / total
            if exp > 0:
                x += (obs - exp) ** 2 / exp
    df = (len(rowsum) - 1) * (len([c for c in colsum if c]) - 1)
    return x, df


def chi2_sf(x, df):
    try:
        from scipy.stats import chi2 as _c
        return float(_c.sf(x, df))
    except ImportError:                      # Wilson-Hilferty, adequate for reporting
        z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
        return 0.5 * math.erfc(z / math.sqrt(2))


def split_notes(note):
    """The rater tool writes tags as leading semicolon-separated tokens.

    Returns (tags actually clicked, remaining free text). A tag word typed into
    the free text but never clicked is NOT a tag -- that distinction is the
    difference between 99 donut images and the 190 word-occurrences you get
    from a naive regex over the whole field.
    """
    parts = [p.strip() for p in note.split(';') if p.strip()]
    tags = {p.lower() for p in parts if p.lower() in TAGS}
    rest = ' '.join(p for p in parts if p.lower() not in TAGS).lower()
    return tags, rest


def mentioned(rest):
    return {t for t in TAGS if re.search(t + r'(s|ed)?\b', rest)}


def main():
    rows = list(csv.DictReader(open(LABELS)))
    ids = [r['image_id'] for r in rows]
    rated = [r for r in rows if r['blinded_call'].strip()]
    n = len(rows)

    print("== integrity ==")
    print("rows                     :", n)
    print("unique ids               :", len(set(ids)))
    print("malformed ids            :", sum(1 for i in ids if not re.fullmatch(r'img_[0-9a-f]{6}', i)))
    print("rated / blank            :", len(rated), "/", n - len(rated))
    first_blank = next((k for k, r in enumerate(rows) if not r['blinded_call'].strip()), n)
    print("rated after first blank  :", sum(1 for r in rows[first_blank:] if r['blinded_call'].strip()))
    print("invalid call values      :", sorted({r['blinded_call'] for r in rated} - set(CALLS)) or 'none')
    print("rated without a date     :", sum(1 for r in rated if not r['date_rated'].strip()))
    print("unrated carrying data    :", sum(1 for r in rows if not r['blinded_call'].strip()
                                            and (r['date_rated'].strip() or r['notes'].strip())))
    if os.path.exists(RATER):
        order = re.findall(r'img_[0-9a-f]{6}',
                           re.search(r'ORDER\s*=\s*\[(.*?)\]', open(RATER).read(), re.S).group(1))
        print("csv order == tool ORDER  :", ids == order)
    if os.path.isdir(PNGDIR):
        png = {f[:-4] for f in os.listdir(PNGDIR) if f.endswith('.png')}
        print("ids with no rendering    :", len(set(ids) - png))

    print("\n== call distribution ==")
    cnt = collections.Counter(r['blinded_call'] for r in rated)
    for c in CALLS:
        print(f"  {c:<12}{cnt[c]:>5}  {100*cnt[c]/len(rated):5.1f}%")

    print("\n== tags ==")
    clicked, prose_only, per_call = collections.Counter(), collections.Counter(), \
        collections.defaultdict(collections.Counter)
    any_tag = both = 0
    for r in rated:
        tags, rest = split_notes(r['notes'])
        extra = mentioned(rest) - tags
        for t in tags:
            clicked[t] += 1
        for t in extra:
            prose_only[t] += 1
        union = tags | extra
        if union:
            any_tag += 1
        if len(union) == 2:
            both += 1
        for t in union:
            per_call[t][r['blinded_call']] += 1
    for t in TAGS:
        print(f"  {t:<7} clicked {clicked[t]:>4}  typed but not clicked {prose_only[t]:>3}"
              f"  total {clicked[t]+prose_only[t]:>4}")
    print(f"  images with any tag {any_tag}, with both {both}")
    for t in TAGS:
        print(f"  {t} by call:", dict(per_call[t]))

    print("\n== order-effect check ==")
    print("Presentation order is random with respect to image content, so a change")
    print("in the calls across that order indicates rater drift, not the pool.")
    B, size = 5, n // 5
    table = []
    print(f"  {'position':<12}" + "".join(f"{c[:5]:>7}" for c in CALLS) + f"{'note%':>8}{'tag%':>7}")
    for b in range(B):
        chunk = rows[b*size:(b+1)*size]
        c = collections.Counter(r['blinded_call'] for r in chunk)
        table.append([c[k] for k in CALLS])
        notes = sum(1 for r in chunk if split_notes(r['notes'])[1])
        tag = sum(1 for r in chunk if split_notes(r['notes'])[0] or mentioned(split_notes(r['notes'])[1]))
        print(f"  {b*size+1:>4}-{(b+1)*size:<7}" + "".join(f"{c[k]:>7}" for k in CALLS)
              + f"{100*notes/len(chunk):>7.1f}%{100*tag/len(chunk):>6.1f}%")
    x, df = chi2(table)
    print(f"  call x quintile: chi2={x:.2f} df={df} p={chi2_sf(x, df):.4f}")

    h = n // 2
    for name, fn in [("uncertain", lambda r: r['blinded_call'] == 'uncertain'),
                     ("any tag", lambda r: bool(split_notes(r['notes'])[0]
                                                or mentioned(split_notes(r['notes'])[1]))),
                     ("free note", lambda r: bool(split_notes(r['notes'])[1]))]:
        a = sum(1 for r in rows[:h] if fn(r))
        b = sum(1 for r in rows[h:] if fn(r))
        x, _ = chi2([[a, h - a], [b, n - h - b]])
        print(f"  {name:<10} first {100*a/h:5.1f}%  second {100*b/(n-h):5.1f}%  "
              f"chi2={x:.2f} p={chi2_sf(x, 1):.4f}")


if __name__ == '__main__':
    main()
