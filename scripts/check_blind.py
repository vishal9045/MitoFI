"""Guard against condition information leaking into the repository.

Golden rule one is that condition identity never appears in a file name, a
folder name, or the working directory. This is easy to violate by accident: an
early metadata export under original filenames did exactly that, and sat in git
for several commits before being noticed.

Run before any push that matters, and before the repository is ever made public.
Exits non-zero if anything is found.

    python3 scripts/check_blind.py
"""
import os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Condition and cell-line tokens used in this project's source filenames.
# Acronyms are matched case-sensitively: a case-insensitive 'CCCP' or 'EV' hits
# random byte sequences in every binary file in the repository.
# Extend these lists rather than relying on them being complete.
TOKENS_CASE = [r'\bCCCP\b', r'\bEV\b', r'\bDMSO\b', r'\bHeLa\b', r'Huh-?7', r'\bHUH-?7\b']
# 'treated' alone is ordinary English ('each image is treated alone') and
# produced only false positives; 'untreated' is specific enough to keep.
TOKENS_ANY = [r'\bmock\b', r'\buntreated\b', r'\bcontrol\b']
# Literature discussion legitimately names drugs; these files are prose, not data.
PROSE = {'notes/fragmentation-field-guide.html', 'notes/blind-pilot-protocol.html',
         'scripts/check_blind.py', 'notes/rating_rubric.md'}
# Independent validation sets live under validation/. They are deliberately NOT
# blind -- the condition of every field is what is being tested -- so condition
# words are expected there. What must never appear there is a pilot image ID:
# condition labels next to a pilot ID is exactly the leak this script exists to
# stop. So that folder trades the word scan for an ID scan.
UNBLINDED = 'validation/'
PILOT_ID = re.compile(r'\bimg_[0-9a-f]{6}\b')
# Files that must never be tracked at all.
FORBIDDEN = [r'_BLIND_KEY', r'tally_sheet', r'removed_by_qc', r'condition_key',
             r'metadata_manifest']


def is_binary(path):
    with open(path, 'rb') as fh:
        return b'\0' in fh.read(8192)


def tracked():
    """Every tracked file, or a hard stop.

    Fails closed. A Git ownership complaint or a missing repository used to make
    this return an empty list, which the scanner then reported as clean -- the
    one failure mode a leak scanner must not have, because a clean report is
    exactly what it produces when it has looked at nothing at all.
    """
    out = subprocess.run(['git', 'ls-files'], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f'cannot list tracked files (git exited {out.returncode}): '
                 f'{out.stderr.strip() or "no message"}\n'
                 f'Refusing to report on a file list this scanner could not read.')
    files = [f for f in out.stdout.splitlines() if f]
    if not files:
        sys.exit('git listed no tracked files at all. Refusing to report clean.')
    return files


def main():
    bad = []
    pats = [re.compile('|'.join(TOKENS_CASE)), re.compile('|'.join(TOKENS_ANY), re.I)]
    forb = re.compile('|'.join(FORBIDDEN), re.I)

    for f in tracked():
        if forb.search(f):
            bad.append((f, 'forbidden file is tracked'))
            continue
        if f in PROSE:
            continue
        if f.startswith(UNBLINDED):
            path = os.path.join(ROOT, f)
            if os.path.exists(path) and not is_binary(path):
                mo = PILOT_ID.search(open(path, encoding='utf-8', errors='ignore').read())
                if mo:
                    bad.append((f, f'pilot image ID {mo.group()!r} in an unblinded '
                                   f'validation folder'))
            continue
        if any(p.search(f) for p in pats):
            bad.append((f, 'condition token in the path'))
        path = os.path.join(ROOT, f)
        if not os.path.exists(path) or os.path.getsize(path) > 5_000_000:
            continue
        try:
            if is_binary(path):
                continue
            text = open(path, encoding='utf-8', errors='ignore').read()
        except OSError:
            continue
        for p in pats:
            mo = p.search(text)
            if mo:
                line = text[:mo.start()].count('\n') + 1
                bad.append((f"{f}:{line}", f"condition token {mo.group()!r} in content"))
                break

    if bad:
        print(f"BLIND LEAK: {len(bad)} finding(s)")
        for where, what in bad:
            print(f"  {where}  --  {what}")
        return 1
    print(f"clean: {len(tracked())} tracked files scanned, binaries skipped, "
          f"no condition tokens outside the prose allowlist")
    return 0


if __name__ == '__main__':
    sys.exit(main())
