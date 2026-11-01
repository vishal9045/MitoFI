"""A small desktop window for scoring a folder of images.

Standard library only: tkinter ships with Python, so this adds no dependency and
runs wherever the package installs.

It deliberately shows the segmentation mask over the image. Every failure found
while developing this index was caught by looking at a mask, not by reading a
metric: a hole-filling rule that passed every summary statistic and still fused
adjacent tubules, a feature set that improved its headline numbers by segmenting
noise. A user who never sees a mask cannot notice the classifier is wrong on
their microscope.

That is why the preview zooms. A network fitted to a 460 pixel thumbnail is not
evidence about a mask drawn on a 1024 pixel image: at that reduction one screen
pixel covers several tubule widths, and the two failures worth catching -- a
mask that fuses neighbouring tubules, and one that traces noise -- are invisible
until the pixels are shown at their own size.

    mitofi gui
"""
import csv
import os
import queue
import threading
import time
import traceback
from collections import OrderedDict

import numpy as np

from . import __version__

PREVIEW_PX = 460            # the preview's minimum size; it grows with the window
FILE_EXT = ('.tif', '.tiff')
# used to preselect a channel, never to decide one silently: the pick is always
# shown, and the preview shows what it produced
MITO_HINTS = ('mito', 'tom20', 'tom70', 'tomm', 'hsp60', 'atp5', 'pdh')
ZOOMS = (1, 2, 4, 8)        # screen pixels per image pixel; None means fit to the panel
VIEWS = 4                   # full-resolution images held in memory at once


def _stem(name):
    stem = os.path.splitext(name)[0]
    return stem[:-4] if stem.endswith('.ome') else stem


def _ids(files):
    """{filename: image_id}, all distinct.

    `a.tif`, `a.tiff` and `a.ome.tif` all reduce to `a`, and that one string was
    the Treeview row id, the saved image_id and the resume key -- so the second
    file overwrote the first everywhere at once, silently. Where a stem is not
    unique, the filename is used instead; only the ambiguous ones get the longer
    name.
    """
    groups = {}
    for f in files:
        groups.setdefault(_stem(f), []).append(f)
    return {f: (_stem(f) if len(g) == 1 else f)
            for g in groups.values() for f in g}


EX_PREFIX = 'excitation '


def _resolve(layout, label, exc=()):
    """Channel index for one file, by name or by laser line. None for one channel.

    Looked up per file rather than fixed once for the folder, because the same
    experiment can export the same dye at different positions. A label of the
    form "excitation 594 nm" matches the laser line instead of the name, which
    matters because names can be wrong: on Leica exports the channel named
    "Mito Tracker Red" was not the mitochondrial one.
    """
    if not layout:
        return None
    if label.startswith(EX_PREFIX):
        nm = float(label[len(EX_PREFIX):].split()[0])
        hits = [i for i, w in enumerate(exc) if w is not None and abs(w - nm) <= 1]
        if len(hits) == 1:
            return hits[0]
        raise ValueError(f'{len(hits) or "no"} channel(s) excited at {nm:g} nm here')
    name = label.split('  [ex ')[0]
    if name in layout:
        return layout.index(name)
    raise ValueError(f'no channel named {name!r} here; this file holds '
                     + ', '.join(layout))


def _ppm(rgb):
    """Encode an HxWx3 uint8 array for tkinter PhotoImage, without Pillow.

    Raw PPM bytes, not base64: Tk's PPM handler does not base64-decode, and
    handing it an encoded string fails with "couldn't recognize image data".
    """
    h, w = rgb.shape[:2]
    return f'P6\n{w} {h}\n255\n'.encode() + rgb.tobytes()


def edges(mask):
    """The mask's own boundary pixels: an outline, so the signal stays visible."""
    e = mask ^ np.roll(mask, 1, axis=0)
    e |= mask ^ np.roll(mask, -1, axis=0)
    e |= mask ^ np.roll(mask, 1, axis=1)
    e |= mask ^ np.roll(mask, -1, axis=1)
    return e & mask


def render(img, edge, lohi=None):
    """Grey image with the outline in green, at whatever resolution it is given.

    The contrast limits are passed in when zooming, so that panning around an
    image does not restretch the grey and make the same mitochondria look
    brighter in one corner than another.
    """
    im = img.astype(np.float32)
    lo, hi = lohi if lohi else (np.percentile(im, 1), np.percentile(im, 99.5))
    g = np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.repeat((g * 255).astype(np.uint8)[..., None], 3, axis=2)
    rgb[edge] = (60, 220, 130)
    return rgb


def _set_icon(root, tk):
    """Put the mitochondrion on the window, and on the Windows taskbar.

    Cosmetic, so every part of it is allowed to fail quietly: an old Tk without
    the PNG reader, or a window manager that ignores icons, must not stop the
    tool from opening.
    """
    from .icon import png
    try:
        root._icons = [tk.PhotoImage(data=png(n)) for n in (64, 32)]
        root.iconphoto(True, *root._icons)
    except Exception:
        return
    try:
        # Windows groups taskbar buttons by application id, and a Python script
        # inherits the interpreter's -- without this the taskbar shows Python's
        # icon next to our window's own
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('mitofi')
    except Exception:
        pass


class App:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.root = root
        root.title(f'MitoFI {__version__} — mitochondrial fragmentation index')
        _set_icon(root, tk)
        root.geometry('1180x760')
        root.minsize(900, 560)

        self.folder = tk.StringVar()
        self.channel = tk.StringVar()
        self.px_um = tk.StringVar()
        self.status = tk.StringVar(value='Choose a folder of images.')
        self.hint = tk.StringVar(value='Channel and pixel size are read from the files. '
                                       'Change them only to override the metadata.')
        self.zoom_lab = tk.StringVar(value='fit')
        self.resume = tk.BooleanVar(value=False)
        self.autosave, self.t0 = None, None
        self.rows, self.photo = [], None
        self.files, self.layouts, self.ids = [], {}, {}
        self.excitations = {}                  # per file: laser line of each channel
        self.src, self.masks = {}, {}          # per image: where it came from, its mask
        self.views = OrderedDict()             # a few full-resolution images, for zooming
        self.current, self.zoom, self.center = None, None, (0.5, 0.5)
        self.follow = True                     # preview trails the run until a row is clicked
        self._auto, self._redraw, self._drag = None, None, None
        self.background_pass = False           # the current run is a haze re-score
        self.black_point_pass = False          # the black-point offer has been made
        from tkinter import messagebox
        self.ask_yes = messagebox.askyesno     # replaced by tests, which have no one to ask
        self.q = queue.Queue()

        top = ttk.Frame(root, padding=10)
        top.pack(fill='x')
        self.pick_btn = ttk.Button(top, text='Choose folder…', command=self.pick)
        self.pick_btn.grid(row=0, column=0)
        ttk.Label(top, textvariable=self.folder, width=58).grid(row=0, column=1, columnspan=5,
                                                               sticky='w', padx=8)
        ttk.Label(top, text='Channel').grid(row=1, column=0, sticky='e', pady=(8, 0))
        self.chan_box = ttk.Combobox(top, textvariable=self.channel, width=36,
                                     state='disabled')
        self.chan_box.grid(row=1, column=1, columnspan=2, sticky='w',
                           padx=(4, 14), pady=(8, 0))
        ttk.Label(top, text='Pixel size µm').grid(row=1, column=3, sticky='e', pady=(8, 0))
        self.px_entry = ttk.Entry(top, textvariable=self.px_um, width=10)
        self.px_entry.grid(row=1, column=4, sticky='w', padx=(4, 14), pady=(8, 0))
        ttk.Label(top, textvariable=self.hint, foreground='#666', wraplength=980,
                  justify='left').grid(row=2, column=0, columnspan=6, sticky='w', pady=(6, 0))

        bar = ttk.Frame(root, padding=(10, 0))
        bar.pack(fill='x')
        self.run_btn = ttk.Button(bar, text='Run', command=self.start)
        self.run_btn.pack(side='left')
        self.save_btn = ttk.Button(bar, text='Save a copy…', command=self.save,
                                   state='disabled')
        self.save_btn.pack(side='left', padx=6)
        self.resume_box = ttk.Checkbutton(bar, text='Resume', variable=self.resume,
                                          state='disabled')
        self.resume_box.pack(side='left', padx=(0, 6))
        self.progress = ttk.Progressbar(bar, mode='determinate', length=260)
        self.progress.pack(side='left', padx=12)
        ttk.Label(bar, textvariable=self.status).pack(side='left')

        body = ttk.Frame(root, padding=10)
        body.pack(fill='both', expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1, minsize=420)
        body.columnconfigure(1, weight=2, minsize=PREVIEW_PX)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky='nsew')
        # Anchors are set on the heading and the column together: a right-aligned
        # number under a centred heading is what makes a table look crooked.
        cols = (('image', 260, 'w'), ('index', 70, 'e'), ('band', 110, 'w'),
                ('conf', 60, 'e'), ('QC', 230, 'w'))
        self.tree = ttk.Treeview(left, columns=[c for c, _, _ in cols], show='headings')
        for name, width, anchor in cols:
            self.tree.heading(name, text=name, anchor=anchor)
            self.tree.column(name, width=width, anchor=anchor, stretch=(name in ('image', 'QC')))
        scroll = ttk.Scrollbar(left, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side='left', fill='both', expand=True)
        scroll.pack(side='left', fill='y')
        self.tree.bind('<<TreeviewSelect>>', self.on_select)

        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky='nsew', padx=(10, 0))
        self.canvas = tk.Canvas(right, width=PREVIEW_PX, height=PREVIEW_PX,
                                bg='#111', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', self.later)          # follow the window
        self.canvas.bind('<MouseWheel>', self.wheel)         # Windows, macOS
        self.canvas.bind('<Button-4>', self.wheel)           # X11
        self.canvas.bind('<Button-5>', self.wheel)
        self.canvas.bind('<ButtonPress-1>', self.grab)
        self.canvas.bind('<B1-Motion>', self.pan)
        self.canvas.bind('<Double-Button-1>', lambda e: self.set_zoom(None))

        zbar = ttk.Frame(right)
        zbar.pack(fill='x', pady=(6, 0))
        ttk.Button(zbar, text='−', width=3,
                   command=lambda: self.step_zoom(-1)).pack(side='left')
        ttk.Button(zbar, text='Fit', width=5,
                   command=lambda: self.set_zoom(None)).pack(side='left', padx=4)
        ttk.Button(zbar, text='+', width=3,
                   command=lambda: self.step_zoom(1)).pack(side='left')
        ttk.Label(zbar, textvariable=self.zoom_lab, width=6).pack(side='left', padx=(8, 0))
        ttk.Label(zbar, text='Scroll to zoom, drag to pan, double-click to fit. '
                            'Segmentation outlined in green.',
                  foreground='#666').pack(side='left')
        root.after(100, self.drain)

    # ---------------------------------------------------------------- actions
    def pick(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(title='Folder of images')
        if d:
            self.folder.set(d)
            self.scan()

    def scan(self):
        """Read every file's metadata before anything is segmented.

        Headers only, so it costs one open per file, and it answers up front the
        two questions that otherwise fail the run halfway through: does the file
        carry a pixel size, and which channel is the mitochondria.
        """
        from .io import channels, excitations, pixel_size_um
        d = self.folder.get()
        if not d or not os.path.isdir(d):
            return
        self.files = sorted(f for f in os.listdir(d) if f.lower().endswith(FILE_EXT))
        if not self.files:
            self.status.set('No TIFFs in that folder.')
            self.chan_box.config(values=(), state='disabled')
            self.channel.set('')
            return
        sizes, self.layouts = [], {}
        self.ids = _ids(self.files)
        for f in self.files:
            path = os.path.join(d, f)
            sizes.append(pixel_size_um(path))
            self.layouts[f] = tuple(channels(path))
            self.excitations[f] = tuple(excitations(path))
        have = sum(v is not None for v in sizes)
        msg = f'{len(self.files)} file(s);  pixel size in {have}/{len(self.files)}'
        if have < len(self.files):
            msg += ' — type one above for the rest'

        # results from an earlier run of this folder: offer to carry them over
        from .cli import _read_done
        self.autosave = os.path.join(d, 'mitofi.csv')
        from .cli import check_manifest
        done = _read_done(self.autosave)
        why = check_manifest(self.autosave, self.run_settings()) if done else None
        if done and not why:
            self.resume_box.config(state='normal')
            self.resume.set(True)
            msg += f';  {len(done)} already scored in mitofi.csv'
        else:
            self.resume_box.config(state='disabled')
            self.resume.set(False)
            if why:
                msg += (';  mitofi.csv came from a different run, '
                        'so it cannot be carried over')
        self.status.set(msg)
        self.set_channels()

    def set_channels(self):
        """Offer the channels by name, and say when the folder is not uniform."""
        layouts = {v for v in self.layouts.values() if v}
        if not layouts:
            self.chan_box.config(values=(), state='disabled')
            self.channel.set('')
            self.hint.set('Single-channel files: nothing to choose.')
            return
        shared = sorted(set.intersection(*(set(v) for v in layouts)))
        multi = [f for f, v in self.layouts.items() if v]

        def line_of(name):
            """The laser line of a named channel, if it is the same in every file."""
            ws = set()
            for f in multi:
                i, ex = self.layouts[f].index(name), self.excitations.get(f, ())
                ws.add(ex[i] if i < len(ex) else None)
            return ws.pop() if len(ws) == 1 and None not in ws else None

        named = [n + (f'  [ex {line_of(n):g} nm]' if line_of(n) else '') for n in shared]
        # laser lines that occur exactly once in every file can be chosen directly
        per_file = []
        for f in multi:
            ws = [w for w in self.excitations.get(f, ()) if w is not None]
            per_file.append({w for w in ws if ws.count(w) == 1})
        lines = sorted(set.intersection(*per_file)) if per_file else []
        by_line = [f'{EX_PREFIX}{w:g} nm' for w in lines]
        values = named + by_line
        self.chan_box.config(values=values, state='readonly')
        pick = next((c for c in named if any(h in c.lower() for h in MITO_HINTS)), None)
        self.channel.set(pick or (values[0] if len(values) == 1 else ''))

        note = []
        if len(layouts) > 1:
            note.append(f'{len(layouts)} different channel layouts in this folder. '
                        f'The channel is matched in each file, by name or by laser '
                        f'line, so its position may differ between them.')
            if not values:
                note.append('No channel name or laser line is common to all of them: '
                            'score them in separate folders.')
        if by_line:
            note.append('Channels are listed by name and by laser line. Names are '
                        'typed at acquisition and can be wrong: if the laser line '
                        'shown for the named channel is not the one your '
                        'mitochondrial stain was excited with, choose the '
                        '"excitation ... nm" entry instead.')
        if any(c.startswith('channel ') for v in layouts for c in v):
            note.append('These files record no channel names, so the choice is by '
                        'position only. Check the preview before trusting the numbers.')
        if pick:
            note.append(f'Picked {pick!r} by name. Change it if that is not the '
                        f'mitochondrial stain.')
        elif shared:
            note.append('Choose the mitochondrial channel.')
        self.hint.set(' '.join(note))

    def run_settings(self):
        """What a carried-over row must have been produced under.

        The channel is a name here and an index in the CLI, so a CSV written by
        one will not silently resume in the other -- which is right: they do not
        mean the same thing.
        """
        from . import __version__
        from .cli import _sha
        from .score import INDEX_MODEL
        from .segment import PIXEL_MODEL
        return {
            'mitofi': __version__,
            'front_end': 'gui',
            'channel_name': self.channel.get() or None,
            'px_um': self.px_um.get().strip() or None,
            'folder': os.path.abspath(self.folder.get()) if self.folder.get() else None,
            'index_model_sha256': _sha(INDEX_MODEL),
            'pixel_model_sha256': _sha(PIXEL_MODEL),
        }

    def freeze(self, on):
        """Lock or release the inputs the running worker was handed."""
        state = 'disabled' if on else 'normal'
        self.pick_btn.config(state=state)
        self.px_entry.config(state=state)
        self.resume_box.config(
            state='disabled' if (on or not (self.autosave
                                            and os.path.exists(self.autosave)))
            else 'normal')
        self.chan_box.config(state='disabled' if on else
                             ('readonly' if self.chan_box['values'] else 'disabled'))

    def start(self):
        if not self.folder.get():
            self.status.set('Choose a folder first.')
            return
        if not self.files:
            self.scan()
        if any(self.layouts.values()) and not self.channel.get():
            self.status.set('Choose the mitochondrial channel first.')
            return
        self.tree.delete(*self.tree.get_children())
        self.rows, self.src, self.masks = [], {}, {}
        self.views.clear()
        self.current, self.follow = None, True
        self.canvas.delete('all')

        from .cli import _read_done
        skip = set()
        if self.resume.get():
            known = set(self.ids.values())
            for r in _read_done(self.autosave):
                if r['image_id'] not in known:
                    continue          # a row from some other input set
                self.rows.append(r)
                skip.add(r['image_id'])
                self.add_tree_row(r)
        todo = [f for f in self.files if self.ids[f] not in skip]
        if not todo:
            self.status.set(f'{len(skip)} already scored; nothing left to do.')
            return
        self.background_pass = self.black_point_pass = False
        self.launch(todo)

    def launch(self, todo, background_um=None, black_point=False):
        """Score `todo` on a worker thread, optionally rescued."""
        self.t0 = time.monotonic()
        self.run_btn.config(state='disabled')
        self.save_btn.config(state='disabled')
        # the worker has copied these; changing them mid-run would point
        # `autosave` at another folder while this run's rows are still queued
        self.freeze(True)
        # Read every widget value here, on the main thread. Tk objects must not
        # be touched from a worker: doing so raises "main thread is not in main
        # loop", sometimes immediately and sometimes only under load.
        params = dict(folder=self.folder.get(), files=todo,
                      layouts=dict(self.layouts), channel=self.channel.get(),
                      ids=dict(self.ids), px_um=self.px_um.get().strip(),
                      excitations=dict(self.excitations),
                      background_um=background_um, black_point=black_point)
        threading.Thread(target=self.work, args=(params,), daemon=True).start()

    def work(self, params):
        try:
            from .io import subtract_background, subtract_floor, to_field
            from .score import score_mask
            from .segment import segment
            folder, files = params['folder'], params['files']
            layouts, label = params['layouts'], params['channel']
            if not files:
                self.q.put(('error', 'No TIFFs in that folder')); return
            override = None
            if params['px_um']:
                from .features import check_px
                override = check_px(params['px_um'])    # 0 or -1 is not "unset"

            self.q.put(('total', len(files)))
            for k, f in enumerate(files, 1):
                iid = params['ids'][f]
                px, note, img, mask, ch = None, '', None, None, None
                try:
                    ch = _resolve(layouts.get(f, ()), label,
                                  params['excitations'].get(f, ()))
                    img, file_px, note = to_field(os.path.join(folder, f), channel=ch)
                    px = file_px if override is None else override
                    if px is None:
                        raise ValueError('no pixel size in the file; enter one above')
                    if params['background_um']:
                        img = subtract_background(img, px, params['background_um'])
                        note = '; '.join(n for n in (
                            note, f"background subtracted, rolling ball "
                                  f"{params['background_um']:g} um") if n)
                    if params['black_point']:
                        img, fl = subtract_floor(img, px, peak=True)
                        note = '; '.join(n for n in (
                            note, f'black point at the cytoplasm peak '
                                  f'({fl:.0f} on the 12-bit scale)') if n)
                    mask = segment(img)
                    r = score_mask(mask, px, img=img)
                except Exception as e:                     # one bad file must not stop the run
                    r, img, mask = dict(qc_pass=False, qc_reason=str(e)), None, None
                r['image_id'] = iid
                r['px_um'] = '' if px is None else px
                r['reduction'] = note
                r['rescue'] = ('black point' if params['black_point'] else
                               'background' if params['background_um'] else '')
                self.q.put(('row', (r, img, mask, os.path.join(folder, f), ch, k)))
            self.q.put(('done', None))
        except Exception:
            self.q.put(('error', traceback.format_exc().strip().splitlines()[-1]))

    # ---------------------------------------------------------------- ui pump
    def drain(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == 'total':
                    self.progress.config(maximum=payload, value=0)
                elif kind == 'row':
                    self.add_row(*payload)
                elif kind == 'done':
                    from .cli import summarise
                    from .score import refusal_kind
                    s = summarise(self.rows)
                    msg = (f"{s['n_images']} images: {s['n_scored']} scored, "
                           f"{s['n_rejected']} rejected")
                    if s['n_rejected']:
                        kinds = [refusal_kind(r.get('qc_reason'))[0]
                                 for r in self.rows if not r.get('qc_pass')]
                        msg += ' (' + ', '.join(f'{k} {kinds.count(k)}'
                                                for k in dict.fromkeys(kinds)) + ')'
                    if s['n_scored']:
                        msg += (f"   index mean {s['mean_index']:.3f}, "
                                f"median {s['median_index']:.3f}")
                    from .cli import _clock, rescue_mix_warning
                    if rescue_mix_warning(self.rows):
                        msg += '   NOTE: rescued and untouched fields mixed, see the rescue column'
                    if self.t0:
                        msg += f'   in {_clock(time.monotonic() - self.t0)}'
                    if self.autosave:
                        self.write(self.autosave)
                        msg += f'   \u2192 {os.path.basename(self.autosave)}'
                    self.status.set(msg)
                    self.run_btn.config(state='normal')
                    self.save_btn.config(state='normal')
                    self.freeze(False)
                    self.offer_background()
                elif kind == 'error':
                    self.status.set(payload)
                    self.run_btn.config(state='normal')
                    self.freeze(False)
        except queue.Empty:
            pass
        self.root.after(100, self.drain)

    def add_row(self, r, img, mask, path, channel, k):
        iid = r['image_id']
        old = next((i for i, x in enumerate(self.rows) if x['image_id'] == iid), None)
        if old is None:
            self.rows.append(r)
        else:
            self.rows[old] = r
        if img is not None:
            self.src[iid] = (path, channel)
            # packed, because a run of 500 masks kept whole is a gigabyte of bools
            self.masks[iid] = (np.packbits(mask), mask.shape)
            self.keep(iid, img, mask)
        self.add_tree_row(r)
        self.progress.config(value=k)
        self.status.set(self.eta(k))
        if self.autosave:
            self.write(self.autosave)          # after every image, so a stopped
                                               # run can be resumed, not repeated
        # the preview trails the run, so a bad segmentation shows while it is
        # still running rather than after every image has been processed
        if self.follow and img is not None:
            self.select(iid)

    def add_tree_row(self, r):
        values = (r['image_id'],
                  '' if r.get('index') is None else f"{r['index']:.3f}",
                  r.get('band', ''),
                  '' if r.get('confidence') is None else f"{r['confidence']:.2f}",
                  'pass' if r['qc_pass'] else r.get('qc_reason', ''))
        if self.tree.exists(r['image_id']):
            self.tree.item(r['image_id'], values=values)
        else:
            self.tree.insert('', 'end', iid=r['image_id'], values=values)

    def offer_background(self):
        """Offer the rescues in turn for every field the gate refused.

        Fields that passed are never touched. First background removal; after
        that pass, a black point at the cytoplasm peak for fields still refused
        (see cli._offer_background for why it is offered and marked). Each is
        asked once per run, and declining one ends the offers.
        """
        from .cli import BLACK_POINT_NOTE, _rescuable, refusal_lines
        from .io import BACKGROUND_UM
        if self.black_point_pass:
            return
        by_id = {v: k for k, v in self.ids.items()}
        files = [by_id[r['image_id']] for r in self.rows
                 if _rescuable(r) and r['image_id'] in by_id]
        if not files:
            return
        head = (f'{len(files)} image(s) were refused by the quality gate:\n\n'
                + '\n\n'.join(l.strip() for l in refusal_lines(self.rows)))
        if not self.background_pass:
            if not self.ask_yes(
                    'Some images were refused',
                    head + f'\n\nImages not processed at acquisition often carry diffuse '
                    f'background, and removing it lets some of them be scored. Subtract the '
                    f'background (rolling ball, {BACKGROUND_UM:g} \u00b5m) and score '
                    f'them again? Images that passed are not touched.\n\n'
                    f'The results will say the background was subtracted.'):
                self.black_point_pass = True               # declined: no more offers
                return
            self.background_pass = True
            self.launch(files, background_um=BACKGROUND_UM)
            return
        self.black_point_pass = True
        if self.ask_yes(
                'Some images are still refused',
                head + '\n\nSet a black point at the cytoplasm level and score them '
                'again? It is what setting the minimum in Brightness/Contrast does, '
                'with the level read from each image. Images that passed are not '
                'touched.\n\n' + BLACK_POINT_NOTE):
            self.launch(files, black_point=True)

    def eta(self, done):
        """How far in, and how much longer, estimated from what is finished."""
        from .cli import _clock
        total = int(self.progress['maximum']) or done
        spent = time.monotonic() - (self.t0 or time.monotonic())
        left = f', ~{_clock(spent / done * (total - done))} left' if done else ''
        carried = len(self.rows) - done
        return (f'{done}/{total}  {_clock(spent)} elapsed{left}'
                + (f'   (+{carried} carried over)' if carried > 0 else ''))

    def write(self, path):
        # via a temporary file: this runs after every image, and a crash or a
        # closed window between truncating and the last row would destroy the
        # whole checkpoint the resume depends on
        from .cli import _save_manifest, _write_rows, write_summary
        _write_rows(path, self.rows)
        _save_manifest(path, self.run_settings())
        stem, ext = os.path.splitext(path)
        write_summary(f'{stem}_summary{ext or ".csv"}', self.rows)

    def select(self, iid):
        """Select a row on the run's behalf, and let the event handler draw it.

        Which row was selected, not a flag set around the call: Tk delivers
        <<TreeviewSelect>> from the event loop, after selection_set has already
        returned, so a flag is always back down by the time the handler reads it
        -- and every automatic selection looked like a click, which switched
        following off on the very first image.
        """
        self._auto = iid
        self.tree.selection_set(iid)
        self.tree.see(iid)

    def on_select(self, _=None):
        sel = self.tree.selection()
        if not sel:
            return
        if sel[0] != self._auto:
            self.follow = False              # the user is steering now
        self.show(sel[0])

    # ------------------------------------------------------------- the preview
    def show(self, iid):
        self.current = iid
        self.center = (0.5, 0.5)
        self.draw()

    def keep(self, iid, img, mask):
        """Hold a few images whole, so zooming does not re-read the disk."""
        im = img.astype(np.float32)
        v = (img, mask, edges(mask),
             (float(np.percentile(im, 1)), float(np.percentile(im, 99.5))))
        self.views[iid] = v
        self.views.move_to_end(iid)
        while len(self.views) > VIEWS:
            self.views.popitem(last=False)
        return v

    def view(self, iid):
        """The image at full resolution, re-read from disk if it has aged out.

        Only the image is re-read; the mask is kept for every image, so nothing
        is ever segmented twice and the picture cannot drift from the score.
        """
        v = self.views.get(iid)
        if v is not None:
            self.views.move_to_end(iid)
            return v
        from .io import to_field
        path, channel = self.src[iid]
        img, _, _ = to_field(path, channel=channel)
        packed, shape = self.masks[iid]
        mask = np.unpackbits(packed)[:shape[0] * shape[1]].reshape(shape).astype(bool)
        return self.keep(iid, img, mask)

    def later(self, *_):
        """Redraw once the window has stopped moving, not on every resize event."""
        if self._redraw:
            self.root.after_cancel(self._redraw)
        self._redraw = self.root.after(80, self.draw)

    def draw(self, *_):
        self._redraw = None
        c = self.canvas
        cw, ch = max(c.winfo_width(), 10), max(c.winfo_height(), 10)
        c.delete('all')
        if self.current is None or self.current not in self.src:
            known = any(r['image_id'] == self.current for r in self.rows)
            c.create_text(cw // 2, ch // 2, fill='#888', width=cw - 20,
                          text=('no preview: this row was carried over from an '
                                'earlier run' if known else 'no preview'))
            return
        c.create_text(cw // 2, ch // 2, fill='#888', text='rendering…')
        c.update_idletasks()
        try:
            img, mask, edge, lohi = self.view(self.current)
        except Exception as e:
            c.delete('all')
            c.create_text(cw // 2, ch // 2, fill='#c66', text=str(e), width=cw - 20)
            return
        h, w = img.shape

        if self.zoom is None:
            step = max(1, int(np.ceil(max(h / ch, w / cw))))
            rgb = render(img[::step, ::step], edges(mask[::step, ::step]), lohi)
            self.zoom_lab.set('fit' if step > 1 else '1:1')
        else:
            z = self.zoom
            vw, vh = min(w, max(8, cw // z)), min(h, max(8, ch // z))
            cx, cy = self.center
            x0 = int(min(max(cx * w - vw / 2, 0), w - vw))
            y0 = int(min(max(cy * h - vh / 2, 0), h - vh))
            # the outline is cut from the whole mask, so the edge of the crop is
            # not mistaken for the edge of an object
            rgb = render(img[y0:y0 + vh, x0:x0 + vw],
                         edge[y0:y0 + vh, x0:x0 + vw], lohi)
            if z > 1:
                rgb = np.repeat(np.repeat(rgb, z, axis=0), z, axis=1)
            self.zoom_lab.set(f'{z}:1')
            self.center = ((x0 + vw / 2) / w, (y0 + vh / 2) / h)

        c.delete('all')
        self.photo = self.tk.PhotoImage(data=_ppm(rgb))       # keep a reference
        c.create_image(cw // 2, ch // 2, image=self.photo)

    # ------------------------------------------------------------ zoom and pan
    def set_zoom(self, z):
        self.zoom = z
        self.draw()

    def step_zoom(self, direction):
        levels = (None,) + ZOOMS
        i = levels.index(self.zoom) if self.zoom in levels else 0
        self.set_zoom(levels[min(max(i + direction, 0), len(levels) - 1)])

    def wheel(self, event):
        up = getattr(event, 'delta', 0) > 0 or getattr(event, 'num', 0) == 4
        self.step_zoom(1 if up else -1)

    def grab(self, event):
        self._drag = (event.x, event.y)

    def pan(self, event):
        if self.zoom is None or self._drag is None or self.current not in self.src:
            return
        dx, dy = event.x - self._drag[0], event.y - self._drag[1]
        self._drag = (event.x, event.y)
        img = self.views[self.current][0] if self.current in self.views else None
        if img is None:
            return
        h, w = img.shape
        cx, cy = self.center
        self.center = (cx - dx / (self.zoom * w), cy - dy / (self.zoom * h))
        self.draw()

    def save(self):
        from tkinter import filedialog
        # start in the image folder: the results belong with the images, not in
        # whatever directory the dialog happened to open in last
        p = filedialog.asksaveasfilename(defaultextension='.csv',
                                         initialdir=self.folder.get() or None,
                                         initialfile='mitofi.csv')
        if not p:
            return
        self.write(p)
        self.status.set(f'Saved {len(self.rows)} rows to {os.path.basename(p)} '
                        f'and the run summary beside it')


def main():
    try:
        import tkinter as tk
    except ImportError:
        raise SystemExit('tkinter is not available in this Python. On Debian or '
                         'Ubuntu: sudo apt install python3-tk')
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
