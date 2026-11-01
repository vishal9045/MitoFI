"""Command line interface.

    mitofi score IMAGES/ --px-um 0.0829
    mitofi score IMAGES/ --px-um-csv sizes.csv -o results.csv
    mitofi score IMAGES/ --masks MASKS/             # masks made elsewhere, gated fully
    mitofi score --masks MASKS/ --px-um 0.0829      # masks alone, size-only gate
    mitofi segment IMAGES/ -o MASKS/
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

from . import __version__
from .features import as_mask
from .io import (BACKGROUND_UM, CalibrationError, channel_by_excitation, pixel_size_um,
                 subtract_background, subtract_floor, to_field)
from .score import OTHER_REFUSAL, REFUSALS, refusal_kind, score_mask
from .segment import segment

COLUMNS = ['image_id', 'qc_pass', 'qc_reason', 'index', 'band', 'confidence',
           'index_unvalidated', 'fg_frac', 'p90_major_um', 'local_contrast',
           'haze', 'local_snr', 'n_objects', 'px_um', 'rescue', 'reduction']


def _read(path, args):
    """Reduce an OME-TIFF to the 2D field the index expects, and its pixel size.

    With --channel-ex the channel is found in each file by its laser line, so a
    folder whose files order or name their channels differently still has the
    same stain scored in every one.
    """
    try:
        ch, how = args.channel, ''
        if getattr(args, 'channel_ex', None) is not None:
            ch = channel_by_excitation(path, args.channel_ex)
            if ch is not None:
                how = f'channel {ch} chosen by excitation at {args.channel_ex:g} nm'
        img, px, note = to_field(path, channel=ch, timepoint=args.timepoint)
        return img, px, '; '.join(n for n in (how, note) if n)
    except ValueError as e:
        sys.exit(f'{os.path.basename(path)}: {e}')


def _file_px(path):
    """A file's own pixel size, or None -- but a refused calibration is fatal.

    "No pixel size" is recoverable: pass --px-um. "This file says its pixels are
    0.1 x 0.2 um" is not, because continuing means ignoring what the file said.
    """
    try:
        return pixel_size_um(path)
    except CalibrationError as e:
        sys.exit(f'{os.path.basename(path)}: {e}')


def _positive_um(text):
    v = float(text)
    if not math.isfinite(v) or v <= 0:
        raise argparse.ArgumentTypeError(
            f'pixel size must be a finite positive number of micrometres, got {text!r}')
    return v


def _pixel_sizes(args, ids):
    """Explicit pixel sizes, if given. Otherwise each file's own OME metadata."""
    if args.px_um_csv:
        with open(args.px_um_csv, newline='') as fh:
            try:
                m = {r['image_id']: _positive_um(r['px_um'])
                     for r in csv.DictReader(fh)}
            except (KeyError, TypeError, ValueError,
                    argparse.ArgumentTypeError) as e:
                sys.exit(f'{args.px_um_csv}: {e}. It needs the columns '
                         f'image_id and px_um, with a positive size in each row.')
        missing = [i for i in ids if i not in m]
        if missing:
            sys.exit(f"px_um missing for {len(missing)} image(s), first: {missing[0]}")
        return m
    if args.px_um is not None:
        return {i: args.px_um for i in ids}
    return {}                    # fall back to the value carried in each file


ID_W = 30          # widest identifier shown before it is elided
BAR = '\u2500'     # box-drawing horizontal


def _fit(text, width):
    """Trim to width, marking the trim so a cut name never reads as a real one."""
    text = str(text)
    return text if len(text) <= width else text[:width - 1] + '\u2026'


def _row(idx, name, px, index, band, conf, note, w):
    return (f'  {idx:>3}  {_fit(name, w):<{w}}  {px:>7}  {index:>6}  '
            f'{band:<11}  {conf:>4}  {note}').rstrip()


def _name_list(names):
    return ', '.join(names[:5]) + (' ...' if len(names) > 5 else '')


def ask_px(missing, total, ask=input):
    """Pixel size typed by the user for files that record none, or None.

    Asks only when someone is there to answer. A script redirected from a file or
    running in a pipeline returns None without printing, so the caller can put
    the whole failure on stderr in one piece rather than splitting it across two
    streams.
    """
    if not sys.stdin.isatty():
        return None
    print(f'{len(missing)} of {total} files record no pixel size: '
          f'{_name_list(missing)}')
    try:
        answer = ask('Pixel size in micrometres for those files '
                     '(Enter to cancel): ').strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not answer:
        return None
    try:
        v = float(answer)
    except ValueError:
        sys.exit(f'{answer!r} is not a number.')
    if v <= 0:
        sys.exit('Pixel size must be greater than zero.')
    return v


def _clock(seconds):
    m, sec = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f'{h}h{m:02d}m' if h else (f'{m}m{sec:02d}s' if m else f'{sec}s')


class Progress:
    """A one-line progress report, on stderr, redrawn in place.

    Segmentation is seconds per image, so a folder of 500 is over an hour during
    which the tool otherwise prints nothing at all: the screen is indistinguishable
    from a hang. It goes to stderr so that stdout stays a clean table that can be
    piped, and it draws nothing at all when stderr is not a terminal, so a log
    file does not fill up with redraws.
    """
    WIDTH = 24

    def __init__(self, total, stream=None):
        self.total = total
        self.stream = sys.stderr if stream is None else stream
        self.on = bool(total) and getattr(self.stream, 'isatty', lambda: False)()
        self.start = time.monotonic()
        self.drawn = 0

    def clear(self):
        """Wipe the line so a finished row can be printed over it."""
        if self.on and self.drawn:
            self.stream.write('\r' + ' ' * self.drawn + '\r')
            self.stream.flush()
            self.drawn = 0

    def show(self, done, label=''):
        if not self.on:
            return
        frac = done / self.total
        filled = int(round(frac * self.WIDTH))
        bar = '\u2588' * filled + '\u2591' * (self.WIDTH - filled)
        spent = time.monotonic() - self.start
        # an estimate from the images already finished; it is honest only after
        # the first one, so nothing is guessed before then
        left = f'~{_clock(spent / done * (self.total - done))} left' if done else 'starting'
        text = f'  {bar} {done}/{self.total}  {_clock(spent)} elapsed, {left}  {label}'
        self.clear()
        self.stream.write(text)
        self.stream.flush()
        self.drawn = len(text)

    def done(self):
        self.clear()


def _read_done(path):
    """Rows already written by an earlier run of the same command.

    Read back as numbers rather than as the strings a CSV gives, because the
    summary treats an index of None as "not scored", and the string 'False' is
    true.
    """
    if not os.path.exists(path):
        return []
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r['qc_pass'] = (r.get('qc_pass') == 'True')
        for k in ('index', 'confidence', 'index_unvalidated', 'px_um',
                  'fg_frac', 'p90_major_um', 'local_contrast'):
            v = r.get(k, '')
            r[k] = float(v) if v not in ('', None) else None
        r['n_objects'] = int(r['n_objects']) if r.get('n_objects') else 0
    return rows


def _atomic(path, write):
    """Write via a temporary file in the same folder, then replace.

    The results CSV is rewritten in full after every image so a stopped run can
    be resumed. Rewriting in place means a Ctrl-C between truncation and the last
    row destroys every result the resume was there to protect -- the failure the
    feature exists to prevent. os.replace is atomic on both POSIX and Windows,
    so the previous file survives intact until the new one is complete.
    """
    tmp = f'{path}.part'
    try:
        with open(tmp, 'w', newline='') as fh:
            write(fh)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _manifest(args):
    """The settings a resumed row must have been produced under.

    Resume used to trust the image_id alone, so changing the pixel size, the
    channel, the timepoint, the masks or the model and re-running with --resume
    silently kept every old row. The carried numbers then described a different
    analysis from the new ones in the same file, with nothing to say so.
    """
    from .segment import PIXEL_MODEL
    from .score import INDEX_MODEL
    return {
        'mitofi': __version__,
        'channel': args.channel,
        'channel_ex': getattr(args, 'channel_ex', None),
        'timepoint': args.timepoint,
        'suffix': args.suffix,
        'mask_suffix': args.mask_suffix if args.masks else None,
        'subtract_background': args.subtract_background,
        'rescue': getattr(args, 'rescue', False),
        'images': os.path.abspath(args.images) if args.images else None,
        'masks': os.path.abspath(args.masks) if args.masks else None,
        'index_model_sha256': _sha(INDEX_MODEL),
        'pixel_model_sha256': _sha(PIXEL_MODEL),
    }


def _manifest_path(out):
    return f'{out}.run.json'


def _ready(path):
    """Make the output's folder, and fail now rather than after the work.

    Checked before the first image is read: discovering an unwritable path after
    an hour of segmentation throws all of it away, and the traceback that used to
    appear there was the one error in this tool that did not read as a sentence.
    """
    parent = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(parent, exist_ok=True)
    except OSError as e:
        sys.exit(f'cannot write {path}: {e.strerror}')
    if os.path.isdir(path):
        sys.exit(f'{path} is a directory; -o takes a file name')
    if not os.access(parent, os.W_OK):
        sys.exit(f'cannot write into {parent}: permission denied')
    return path


def _inputs(folder, suffix):
    """(id, path) for every file ending in `suffix`, with the suffix stripped.

    `suffix` is one string. Iterating it as a sequence of suffixes matches single
    characters instead, which silently strips one letter from every identifier.
    """
    if not isinstance(suffix, str):
        raise TypeError('suffix must be a single string')
    if not suffix:
        sys.exit('the suffix cannot be empty: stripping it leaves every file '
                 'with the same blank identifier')
    if not os.path.isdir(folder):
        sys.exit(f'not a directory: {folder}')
    low = suffix.lower()
    out = []
    for f in sorted(os.listdir(folder)):
        # case-insensitive, because .TIF from a Windows export is the same file
        # type as .tif; directories are skipped, because a folder named
        # something.tif is not an image
        if not f.lower().endswith(low):
            continue
        if os.path.isdir(os.path.join(folder, f)):
            continue
        out.append((f[:-len(suffix)], os.path.join(folder, f)))
    return out


def cmd_segment(args):
    import tifffile
    items = _inputs(args.images, args.suffix)
    if not items:
        sys.exit(f'no files ending {args.suffix} in {args.images}')
    # default output next to the input, not in whatever directory the shell
    # happens to sit in
    args.out = args.out or os.path.join(args.images, 'masks')
    os.makedirs(args.out, exist_ok=True)
    bar = Progress(len(items))
    bar.show(0)
    for k, (iid, path) in enumerate(items, 1):
        img, file_px, _ = _read(path, args)
        size = args.px_um if args.px_um is not None else file_px
        if args.subtract_background:
            if size is None:
                sys.exit(f'{iid}: background subtraction needs a pixel size; '
                         f'pass --px-um')
            img = subtract_background(img, size, args.subtract_background)
        mask = segment(img, jobs=args.jobs)
        # carry the scale onto the mask, so `mitofi score --masks` on this
        # folder does not need to be told the pixel size all over again
        cal = ({'resolution': (1.0 / size, 1.0 / size),
                'metadata': {'unit': 'um'}} if size else {})
        tifffile.imwrite(os.path.join(args.out, iid + '_mask.tif'),
                         mask.astype(np.uint8), compression='zlib',
                         imagej=bool(size), **cal)
        bar.clear()
        print(f'  [{k}/{len(items)}] {iid}  foreground {mask.mean():.2%}', flush=True)
        bar.show(k, iid)
    bar.done()
    print(f'{len(items)} masks -> {os.path.abspath(args.out)}')


def summarise(rows):
    """Aggregate over the images that passed QC.

    Rejected images are excluded rather than counted as zero: the tool declined
    to score them, and averaging in a number it refused to report would be
    inventing data. Their count is carried so the exclusion is visible.
    """
    ok = [r for r in rows if r.get('qc_pass') and r.get('index') is not None]
    v = np.array([r['index'] for r in ok], dtype=float)
    bands = [r.get('band') for r in ok]
    out = {'n_images': len(rows), 'n_scored': len(ok), 'n_rejected': len(rows) - len(ok)}
    if len(ok):
        out.update(mean_index=round(float(v.mean()), 4),
                   median_index=round(float(np.median(v)), 4),
                   sd_index=round(float(v.std(ddof=1)), 4) if len(v) > 1 else '',
                   min_index=round(float(v.min()), 4),
                   max_index=round(float(v.max()), 4))
    for b in ('fragmented', 'mixed', 'elongated'):
        out[f'n_{b}'] = bands.count(b)
    for k in ('background', 'black point'):
        out['n_rescued_' + k.replace(' ', '_')] = sum(r.get('rescue') == k for r in ok)
    # every kind always present, so summaries from different runs line up
    why = [refusal_kind(r.get('qc_reason'))[0] for r in rows
           if not (r.get('qc_pass') and r.get('index') is not None)]
    for name, _ in list(REFUSALS.values()) + [OTHER_REFUSAL]:
        out['rejected_' + name.replace(' ', '_')] = why.count(name)
    return out


def refusal_lines(rows):
    """One line per kind of refusal present: count, name, what it means."""
    kinds = [refusal_kind(r.get('qc_reason')) for r in rows if not r.get('qc_pass')]
    order = list(REFUSALS.values()) + [OTHER_REFUSAL]
    return [f'{kinds.count(k):>4}  {k[0]}: {k[1]}' for k in order if k in kinds]


def _write_rows(path, rows):
    def go(fh):
        w = csv.DictWriter(fh, fieldnames=COLUMNS, restval='',
                           extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    _atomic(path, go)


def write_summary(path, rows):
    s = summarise(rows)

    def go(fh):
        w = csv.DictWriter(fh, fieldnames=list(s))
        w.writeheader(); w.writerow(s)
    _atomic(path, go)
    return s


def _save_manifest(out, run):
    _atomic(_manifest_path(out), lambda fh: json.dump(run, fh, indent=1, sort_keys=True))


def check_manifest(out, run):
    """None when the rows already in `out` were produced under `run`, else why not.

    Shared by the CLI and the GUI because they write the same mitofi.csv into the
    same folder: a run resumed in one must be checked the same way in the other.
    """
    mpath = _manifest_path(out)
    if not os.path.exists(mpath):
        return (f'{os.path.basename(out)} exists but '
                f'{os.path.basename(mpath)} does not, so there is no record of '
                f'how those rows were produced and no way to tell whether they '
                f'match this run.')
    try:
        with open(mpath) as fh:
            was = json.load(fh)
    except (OSError, ValueError) as e:
        return f'cannot read {os.path.basename(mpath)}: {e}'
    changed = [k for k in run if was.get(k) != run[k]]
    if changed:
        return ('the previous rows were produced under different settings:\n'
                + '\n'.join(f'    {k}: {was.get(k)!r} -> {run[k]!r}'
                             for k in changed))
    return None


def _verify_resume(done, args, run, ids, px, paired):
    """The carried rows that still describe this run. Refuses if the run changed.

    Three ways an old row can be wrong, all of them silent before this: it was
    produced under different settings or a different model; it belongs to some
    other input set entirely; or the pixel size this run would use for it is not
    the one recorded against it.
    """
    why = check_manifest(args.out, run)
    if why:
        sys.exit(f'{why}\nResuming would mix two analyses in one file. Re-run '
                 f'without --resume, or delete {os.path.basename(args.out)}.')

    known = set(ids)
    keep, alien, rescaled = [], [], []
    for r in done:
        iid = r['image_id']
        if iid not in known:
            alien.append(iid)
            continue
        want = px.get(iid)
        if want is None:
            src = paired.get(iid)
            if src is None and args.masks:
                src = dict(_inputs(args.masks, args.mask_suffix)).get(iid)
            elif src is None:
                src = dict(_inputs(args.images, args.suffix)).get(iid)
            want = _file_px(src) if src else None
        got = r.get('px_um')
        if want is not None and got is not None and abs(got - want) > 1e-9:
            rescaled.append(iid)
            continue
        keep.append(r)
    if alien:
        print(f'resume: dropping {len(alien)} row(s) not in this input set: '
              f'{_name_list(alien)}')
    if rescaled:
        print(f'resume: re-scoring {len(rescaled)} row(s) whose pixel size '
              f'changed: {_name_list(rescaled)}')
    return keep


def _score_one(iid, path, args, px, paired, background_um=None, black_point=False):
    """Read one field, optionally rescue it, segment and score it."""
    mask = None
    if args.masks:
        import tifffile
        a = tifffile.imread(path)
        if a.ndim != 2:
            sys.exit(f'{iid}: a mask must be 2D, got shape {a.shape}')
        try:
            as_mask(a)
        except ValueError as e:
            sys.exit(f'{iid}: {e}')
        # a mask written by `mitofi segment` carries its own scale; the
        # loop used to hardcode None here, so the preflight found a pixel
        # size and the loop then exited saying the file recorded none
        mask, file_px, note = as_mask(a), _file_px(path), ''
        img = None
        if paired:
            img, file_px, note = _read(paired[iid], args)
            if img.shape != mask.shape:
                sys.exit(f'{iid}: mask is {mask.shape} but its image is '
                         f'{img.shape}; they must be the same field')
    else:
        img, file_px, note = _read(path, args)
    size = px.get(iid, file_px)
    if size is None:
        sys.exit(f'{iid}: no pixel size. The file records none, so pass '
                 f'--px-um or --px-um-csv. The index is in physical units '
                 f'and is meaningless without it.')
    if background_um and img is not None:
        img = subtract_background(img, size, background_um)
        note = '; '.join(n for n in (note, f'background subtracted, rolling '
                                           f'ball {background_um:g} um') if n)
    if black_point and img is not None:
        img, fl = subtract_floor(img, size, peak=True)
        note = '; '.join(n for n in (note, f'black point at the cytoplasm peak '
                                           f'({fl:.0f} on the 12-bit scale)') if n)
    if mask is None:
        mask = segment(img, jobs=args.jobs)
    r = score_mask(mask, size, img=img)
    r.update(image_id=iid, px_um=size, reduction=note,
             rescue='black point' if black_point else 'background' if background_um else '')
    return r


def _print_result(k, r, w):
    size = f"{r['px_um']:.4f}"
    if r['qc_pass']:
        print(_row(k, r['image_id'], size, f"{r['index']:.3f}", r['band'],
                   f"{r['confidence']:.2f}", '', w), flush=True)
    else:
        print(_row(k, r['image_id'], size, '\u2014', '\u2014', '\u2014',
                   r['qc_reason'], w), flush=True)


def _rescuable(r):
    """A refused field one of the rescues might let through.

    Coarse pixels and unreadable files are not: no change to brightness fixes
    either, so offering would only spend time.
    """
    return (not r['qc_pass'] and refusal_kind(r.get('qc_reason'))[0]
            not in ('pixels too coarse', 'could not be read'))


# What the black point does to the scale, measured on pixel classifier v6
# (validation/rescue_options_2026-10, rescue_recovery_2026-10). One sentence,
# shown by the CLI and the GUI, so the two cannot drift apart.
BLACK_POINT_NOTE = ('It keeps the order between fields that all had it, but reads '
                    'about 0.2 higher than untouched fields (median +0.21 to +0.24 on '
                    'fields that already pass), so rows are marked "black point" in '
                    'the rescue column: compare them with each other, not with '
                    'untouched fields.')


def rescue_mix_warning(rows):
    """A warning when scored fields mix untouched and rescued ones, else ''.

    A rescued field went through processing the others did not. Comparing it with
    untouched fields can turn processing into an apparent difference.
    """
    ok = [r for r in rows if r.get('qc_pass') and r.get('index') is not None]
    kinds = [r.get('rescue') or '' for r in ok]
    n_bp, n_bg = kinds.count('black point'), kinds.count('background')
    if not (n_bp or n_bg) or (n_bp + n_bg) == len(ok):
        return ''
    msg = (f'{n_bp + n_bg} of {len(ok)} scored fields were rescued '
           f'(background {n_bg}, black point {n_bp}); the rest were not.')
    if n_bp:
        msg += (' Black-point rows read about 0.2 higher than untouched ones: compare '
                'conditions within the same rescue, or check that a difference holds '
                'without the black-point rows.')
    else:
        msg += ' Check that a difference between conditions holds without the rescued rows.'
    return msg


RESCUES = (
    ('background', 'Removing diffuse background with a {bg:g} um rolling ball lets '
                   'some be scored.',
     'Subtract the background from those {n} and score them again? [y/N] '),
    ('black point', 'Setting a black point at the cytoplasm level -- what setting the '
                    'minimum in Fiji\'s Brightness/Contrast does, with the level read '
                    'from each image -- lets many more be scored. ' + BLACK_POINT_NOTE,
     'Set the black point on those {n} and score them again? [y/N] '),
)


def _offer_background(rows, paths, args, px, paired, w, ask=input):
    """Offer the rescues, in turn, for every field the gate refused.

    Fields that passed are never touched. First background removal; then, for
    fields still refused, a black point at the cytoplasm peak. Images not
    processed at acquisition often carry diffuse or grainy cytoplasmic signal
    that makes the gate refuse them, and correcting it is the user's call, not
    the tool's to make silently -- so it lists the refused fields and asks.
    --rescue says yes to both in advance.

    The black point moves readable fields (D23): with pixel classifier v6 it
    raised the index by a median 0.21 on readable development fields and 0.24 on
    readable Leica series, and on fields damaged on purpose it overshot the clean
    value by +0.04 to +0.29. It is offered (D24) because, on the images that need
    it, the rescued fields ranked conditions as expected; the "rescue" column lets
    every result be checked without them. Denoising
    and brightness matching were tried and rejected.
    """
    for step, why, question in RESCUES:
        if step == 'background' and args.subtract_background:
            continue                                   # every field already had it
        todo = [i for i, r in enumerate(rows) if _rescuable(r) and r['image_id'] in paths]
        if not todo:
            return
        print(f'\n{len(todo)} field(s) are refused by the quality gate. '
              + why.format(bg=BACKGROUND_UM))
        if not getattr(args, 'rescue', False):
            if not sys.stdin.isatty():
                print('Re-run with --rescue to apply the rescues to refused fields.')
                return
            try:
                yes = ask(question.format(n=len(todo))).strip().lower() in ('y', 'yes')
            except (EOFError, KeyboardInterrupt):
                return
            if not yes:
                return
        for k, i in enumerate(todo, 1):
            iid = rows[i]['image_id']
            rows[i] = _score_one(iid, paths[iid], args, px, paired,
                                 BACKGROUND_UM if step == 'background' else None,
                                 black_point=step == 'black point')
            _print_result(k, rows[i], w)
            _write_rows(args.out, rows)


def cmd_score(args):
    if args.masks:
        items = _inputs(args.masks, args.mask_suffix)
        what = 'mask'
    else:
        items = _inputs(args.images, args.suffix)
        what = 'image'
    if not items:
        sys.exit('no input files found')
    args.out = args.out or os.path.join(args.masks or args.images, 'mitofi.csv')
    stem, ext = os.path.splitext(args.out)
    spath = args.summary_out or f'{stem}_summary{ext or ".csv"}'
    if os.path.abspath(spath) == os.path.abspath(args.out):
        sys.exit(f'-o and --summary-out are the same file ({args.out}). The '
                 f'summary would overwrite the per-image results; give them '
                 f'different names.')
    _ready(args.out)
    _ready(spath)

    # BLK-01: masks alone cannot be checked for local contrast, so the gate falls
    # back to object size -- the rule that was rejecting genuinely fragmented
    # cells. Given the images too, the mask path is gated like every other.
    paired = {}
    if args.masks:
        if args.images:
            paired = dict(_inputs(args.images, args.suffix))
            missing = [i for i, _ in items if i not in paired]
            if missing:
                sys.exit(f'no image ending {args.suffix} for {len(missing)} mask(s): '
                         f'{_name_list(missing)}')
        else:
            print('note: masks without their images. Local contrast cannot be '
                  'measured, so the gate falls back to object size alone, which '
                  'rejects genuinely fragmented cells. Give the image folder as '
                  'well to avoid that.')
    all_paths = dict(items)
    ids = [i for i, _ in items]
    px = _pixel_sizes(args, ids)

    run = _manifest(args)
    done_rows = []
    if args.resume:
        done_rows = _read_done(args.out)
        if done_rows:
            done_rows = _verify_resume(done_rows, args, run, ids, px, paired)
        have = {r['image_id'] for r in done_rows}
        items = [(i, p) for i, p in items if i not in have]
        print(f'resuming: {len(have)} already in {os.path.basename(args.out)}, '
              f'{len(items)} to go')
        if not items:
            # the summary is an output of this run, not a by-product of the loop:
            # a fully resumed run used to return here leaving it missing
            _write_rows(args.out, done_rows)
            write_summary(spath, done_rows)
            _save_manifest(args.out, run)
            print(f'nothing left to do\n  {os.path.abspath(args.out)}\n'
                  f'  {os.path.abspath(spath)}')
            return
    _save_manifest(args.out, run)
    # say where the scale comes from. The index is in physical units, so a wrong
    # or silently defaulted pixel size changes every number without any sign.
    if args.px_um_csv:
        print(f'pixel size: from {args.px_um_csv}')
    elif args.px_um is not None:
        print(f'pixel size: {args.px_um} um, given on the command line')
    else:
        print('pixel size: read from each file')
        # Check every file before segmenting any of them. Reading the metadata
        # costs nothing, and discovering a missing scale on image 80 of 100 would
        # otherwise throw away all the work done up to that point.
        # when masks are paired with their images, the calibration lives on the
        # image: a mask written as a plain TIFF carries none
        missing = [i for i, path in items
                   if _file_px(paired.get(i, path)) is None]
        if missing:
            typed = ask_px(missing, len(items))
            if typed is None:
                sys.exit(f'error: {len(missing)} of {len(items)} files record no '
                         f'pixel size: {_name_list(missing)}\n'
                         f'The index is in physical units and is meaningless '
                         f'without one. Pass --px-um for the whole folder, or '
                         f'--px-um-csv for one value per image.')
            # fill only the gaps: files that carry their own calibration keep it
            px.update({i: typed for i in missing})
            print(f'using {typed} um/px for those {len(missing)} file(s)')

    w = min(ID_W, max([len(i) for i, _ in items] + [5]))
    header = _row('#', 'image', 'um/px', 'index', 'band', 'conf', 'notes', w)
    rule = '  ' + BAR * (len(header) - 2)
    print()
    print(header)
    print(rule)

    rows = list(done_rows)
    bar = Progress(len(items))
    bar.show(0)
    for k, (iid, path) in enumerate(items, 1):
        r = _score_one(iid, path, args, px, paired, args.subtract_background)
        rows.append(r)
        bar.clear()
        _print_result(k, r, w)
        bar.show(k, iid)

        # written as we go, so an interrupted run can be resumed rather than
        # repeated: an hour of segmentation is too much to lose to one Ctrl-C
        _write_rows(args.out, rows)
    bar.done()

    _offer_background(rows, all_paths, args, px, paired, w)

    s = write_summary(spath, rows)

    print(rule)
    lab = f'{len(rows)} {what}s'
    print(f"  {lab:<{w + 5}}{s['n_scored']} scored, {s['n_rejected']} rejected")
    if s['n_scored']:
        sd = f"{s['sd_index']:.3f}" if s['sd_index'] != '' else '\u2014'
        print(f"  {'index':<{w + 5}}mean {s['mean_index']:.3f}   "
              f"median {s['median_index']:.3f}   sd {sd}   "
              f"range {s['min_index']:.3f}\u2013{s['max_index']:.3f}")
        print(f"  {'bands':<{w + 5}}" + '   '.join(f"{b} {s['n_' + b]}" for b in
                                                  ('fragmented', 'mixed', 'elongated')))
        sizes = sorted({r['px_um'] for r in rows})
        if len(sizes) > 1:
            print(f"  {'pixel sizes':<{w + 5}}{len(sizes)} different in this folder, "
                  f'{min(sizes):.4f} to {max(sizes):.4f} um')
        if s['n_rejected']:
            print(f"  {'':<{w + 5}}averages cover the scored images only")
        if s['n_rejected'] > s['n_scored']:
            print(f"  {'':<{w + 5}}WARNING: more {what}s were rejected than scored. "
                  f"These averages describe the survivors, which is not the same "
                  f"as describing the experiment.")
    if s['n_rejected']:
        print(f"\n  why {s['n_rejected']} {what}(s) were refused:")
        for line in refusal_lines(rows):
            print('  ' + line)
    mix = rescue_mix_warning(rows)
    if mix:
        print(f'\n  NOTE: {mix}')
    # absolute, because a relative -o lands in the shell's working directory,
    # which is rarely the folder the images came from
    print()
    print(f'  {os.path.abspath(args.out)}')
    print(f'  {os.path.abspath(spath)}')


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='mitofi',
        description='Mitochondrial fragmentation index: 0 = elongated network, '
                    '1 = fragmented. Measures morphology; it is not an assay for '
                    'any treatment.')
    p.add_argument('--version', action='version', version=f'mitofi {__version__}')
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp):
        sp.add_argument('--suffix', default='.tif', help='input image suffix')
        sp.add_argument('--px-um', type=_positive_um,
                        help='pixel size in micrometres; overrides the file')
        sp.add_argument('--px-um-csv', help='CSV with columns image_id,px_um')
        pick = sp.add_mutually_exclusive_group()
        pick.add_argument('--channel', type=int,
                          help='which channel holds the mitochondria, 0-based')
        pick.add_argument('--channel-ex', type=_positive_um, metavar='NM',
                          help='choose the mitochondrial channel in each file by '
                               'its excitation wavelength in nm, e.g. 594; safer '
                               'than --channel when files order channels '
                               'differently, and than names, which can be wrong')
        sp.add_argument('--timepoint', type=int, default=0)
        sp.add_argument('--jobs', type=int,
                        help='cores the classifier votes on; default is all of them')
        sp.add_argument('--subtract-background', nargs='?', const=BACKGROUND_UM,
                        type=_positive_um, metavar='UM',
                        help='remove diffuse background with a rolling ball of '
                             f'this radius in micrometres (default {BACKGROUND_UM:g}) '
                             'before segmenting; for images not processed at '
                             'acquisition')
        if sp.prog.endswith('score'):
            sp.add_argument('--rescue', action='store_true',
                            help='for fields the gate refuses only: remove the '
                                 'background and score again, then set a black '
                                 'point at the cytoplasm level for any still '
                                 'refused; marked in the "rescue" column')

    s = sub.add_parser('segment', help='segment images, write masks')
    s.add_argument('images')
    s.add_argument('-o', '--out',
                   help='output folder; defaults to masks/ inside the image folder')
    common(s)
    s.set_defaults(func=cmd_segment)

    g = sub.add_parser('gui', help='open a window: choose a folder, run, look at the masks')
    g.set_defaults(func=lambda a: __import__('mitofi.gui',
                                             fromlist=['main']).main())

    c = sub.add_parser('score', help='segment (or take masks) and write the index')
    c.add_argument('images', nargs='?', help='folder of images to segment and score')
    c.add_argument('--masks', help='folder of existing binary masks, skips segmentation; '
                                   'give the image folder as well so the quality gate '
                                   'can still measure local contrast')
    c.add_argument('--mask-suffix', default='.tif')
    c.add_argument('-o', '--out',
                   help='output CSV; defaults to mitofi.csv in the input folder')
    c.add_argument('--resume', action='store_true',
                   help='skip images already present in the output CSV')
    c.add_argument('--summary-out',
                   help='where to write the aggregate row; defaults to <out>_summary.csv')
    common(c)
    c.set_defaults(func=cmd_score)

    a = p.parse_args(argv)
    if a.cmd == 'gui':
        return a.func(a)
    if a.cmd == 'score':
        if not a.images and not a.masks:
            p.error('give a folder of images, or --masks')
        # pixel size may also come from each file's OME metadata; that is
        # checked per file, so it is not required on the command line
    a.func(a)


if __name__ == '__main__':
    main()
