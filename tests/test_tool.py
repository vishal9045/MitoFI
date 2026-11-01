"""Checks that fail if the tool's logic breaks. Run with: python tests/test_tool.py"""
import os
import sys
import tempfile

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mitofi import features, qc, score_mask          # noqa: E402
from mitofi.cli import _inputs                        # noqa: E402
from mitofi.io import pixel_size_um, to_field         # noqa: E402
from mitofi.score import MIN_P90_MAJOR_UM             # noqa: E402


def bar_field(n=10, length=40, width=4, px=0.1):
    m = np.zeros((400, 400), bool)
    for k in range(n):
        y = 20 + k * 36
        m[y:y + width, 60:60 + length] = True
    return m, px


def test_physical_units_are_scale_free():
    """The same real object at two pixel sizes must measure the same area."""
    a = np.pad(np.ones((40, 4), bool), 20)
    b = np.pad(np.ones((80, 8), bool), 40)
    fa = features(a, None, 0.1)
    fb = features(b, None, 0.05)
    assert abs(fa['med_area_um2'] - fb['med_area_um2']) < 1e-9, (fa, fb)


def test_gate_rejects_speckle_and_keeps_tubules():
    rng = np.random.default_rng(0)
    speckle = np.zeros((400, 400), bool)
    for _ in range(400):
        y, x = rng.integers(5, 395, 2)
        speckle[y:y + 4, x:x + 4] = True
    f = features(speckle, None, 0.1)
    ok, why = qc(f)
    assert not ok and 'speckle' in why, (ok, why)

    m, px = bar_field()
    f = features(m, None, px)
    assert qc(f)[0], qc(f)


def test_gate_rejects_collapsed_and_empty():
    tiny = np.zeros((400, 400), bool)
    tiny[0:6, 0:6] = True
    assert not qc(features(tiny, None, 0.1))[0]
    assert not qc(features(np.zeros((100, 100), bool), None, 0.1))[0]


def test_rejected_image_gets_no_index():
    tiny = np.zeros((400, 400), bool)
    tiny[0:6, 0:6] = True
    r = score_mask(tiny, 0.1)
    assert r['qc_pass'] is False
    assert r.get('index') is None, r
    assert r['band'] == 'qc_fail'


def round_field(radius_px, px=0.1, step=40):
    yy, xx = np.mgrid[:400, :400]
    m = np.zeros((400, 400), bool)
    for cy in range(40, 380, step):
        for cx in range(40, 380, step):
            m |= np.hypot(yy - cy, xx - cx) < radius_px
    return m, px


def test_index_is_in_range_and_ordered():
    """Round objects must score higher on the fragmentation scale than tubules."""
    tub, px = bar_field()
    rt = score_mask(tub, px)
    assert rt['qc_pass'], rt
    assert 0.0 <= rt['index'] <= 1.0

    # 15 px radius at 0.1 um = 3 um across, a plausible rounded mitochondrion
    frag, _ = round_field(15, px)
    rf = score_mask(frag, px)
    assert rf['qc_pass'], rf
    assert rf['index'] > rt['index'], (rf['index'], rt['index'])


def test_gate_rejects_a_field_of_very_small_objects():
    """Documented behaviour, not a bug: a field whose objects are all about 1 um
    is rejected as speckle. A severely fragmented cell and a broken mask look
    alike, so the gate discards some real biology rather than invent a number."""
    frag, px = round_field(5)          # 1 um across
    r = score_mask(frag, px)
    assert not r['qc_pass'] and 'speckle' in r['qc_reason'], r


def test_suffix_is_one_string_not_a_character_sequence():
    d = tempfile.mkdtemp()
    for n in ('a.ome.tif', 'b.ome.tif', 'readme.txt'):
        open(os.path.join(d, n), 'w').close()
    assert [i for i, _ in _inputs(d, '.ome.tif')] == ['a', 'b']
    assert [i for i, _ in _inputs(d, '.tif')] == ['a.ome', 'b.ome']


def _write_ome(path, axes, shape, px=0.0829, unit='\u00b5m'):
    rng = np.random.default_rng(0)
    meta = {'axes': axes}
    if px:
        meta.update(PhysicalSizeX=px, PhysicalSizeXUnit=unit)
    tifffile.imwrite(path, (rng.random(shape) * 300).astype('uint16'), metadata=meta)


def test_ome_reduction_handles_real_acquisition_layouts():
    d = tempfile.mkdtemp()
    for axes, shape in [('YX', (64, 64)), ('ZYX', (5, 64, 64)),
                        ('ZCYX', (5, 3, 64, 64)), ('TZCYX', (2, 5, 3, 64, 64))]:
        p = os.path.join(d, f'{axes}.ome.tif')
        _write_ome(p, axes, shape)
        img, px, note = to_field(p, channel=1 if 'C' in axes else None)
        assert img.ndim == 2, (axes, img.shape)
        assert img.shape[-2:] == (64, 64), (axes, img.shape)
        assert abs(px - 0.0829) < 1e-9, (axes, px)
        assert note


def test_multichannel_file_refuses_to_guess_a_channel():
    p = os.path.join(tempfile.mkdtemp(), 'c.ome.tif')
    _write_ome(p, 'CYX', (3, 64, 64))
    try:
        to_field(p)
    except ValueError as e:
        assert 'channel' in str(e).lower(), e
    else:
        raise AssertionError('picked a channel without being told which')


def test_pixel_size_units_are_converted():
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'nm.ome.tif')
    _write_ome(p, 'YX', (32, 32), px=82.9, unit='nm')
    assert abs(pixel_size_um(p) - 0.0829) < 1e-9, pixel_size_um(p)
    q = os.path.join(d, 'none.ome.tif')
    _write_ome(q, 'YX', (32, 32), px=None)
    assert pixel_size_um(q) is None


def test_preview_encoding_is_what_tk_accepts():
    """Tk takes raw PPM bytes, not base64. Getting this wrong shows an empty
    preview panel while every number on screen stays correct."""
    from mitofi.gui import _ppm, edges, render
    img = (np.random.default_rng(0).random((300, 300)) * 3000).astype('uint16')
    mask = np.zeros((300, 300), bool)
    mask[80:200, 80:95] = True
    step = 1
    rgb = render(img[::step, ::step], edges(mask[::step, ::step]))
    assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8
    assert (rgb[:, :, 1] == 220).any(), 'mask outline missing from the preview'

    data = _ppm(rgb)
    assert isinstance(data, bytes) and data.startswith(b'P6\n'), data[:12]
    h, w = rgb.shape[:2]
    assert data == f'P6\n{w} {h}\n255\n'.encode() + rgb.tobytes()

    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception:
        return                      # no display in CI; the byte-level checks stand
    root.withdraw()
    try:
        p = tk.PhotoImage(data=data)
        assert (p.width(), p.height()) == (w, h), (p.width(), p.height())
    finally:
        root.destroy()


def test_channel_is_chosen_by_name_not_by_position():
    """The same dye can sit at a different index in two files from one experiment.

    An earlier version of this pipeline took a fixed stack position and scored
    the wrong stain in a fifth of the pilot. The GUI must resolve the channel per
    file, by name.
    """
    from mitofi.gui import MITO_HINTS, _resolve
    from mitofi.io import channels
    d = tempfile.mkdtemp()
    a = np.zeros((3, 2, 16, 16), 'uint16')
    for name, order in (('x.ome.tif', ['DAPI', 'MitoTracker', 'Alexa 647']),
                        ('y.ome.tif', ['MitoTracker', 'DAPI', 'Alexa 647'])):
        tifffile.imwrite(os.path.join(d, name), a,
                         metadata={'axes': 'CZYX', 'Channel': {'Name': order}})
    lay = {f: tuple(channels(os.path.join(d, f))) for f in ('x.ome.tif', 'y.ome.tif')}
    assert _resolve(lay['x.ome.tif'], 'MitoTracker') == 1
    assert _resolve(lay['y.ome.tif'], 'MitoTracker') == 0     # same dye, other slot
    assert _resolve((), 'MitoTracker') is None                # single channel
    try:
        _resolve(lay['x.ome.tif'], 'GFP')
    except ValueError as e:
        assert 'no channel named' in str(e), e
    else:
        raise AssertionError('a missing channel must refuse, not fall back to 0')
    assert any(h in 'mitotracker' for h in MITO_HINTS)        # the preselect hint


def test_unnamed_channels_are_reported_as_positions():
    """A file with no channel names must still say how many it has, and must not
    pretend the placeholders identify anything."""
    from mitofi.io import channels
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'u.ome.tif')
    tifffile.imwrite(p, np.zeros((3, 16, 16), 'uint16'), metadata={'axes': 'CYX'})
    assert channels(p) == ['channel 0', 'channel 1', 'channel 2']
    flat = os.path.join(d, 'flat.tif')
    tifffile.imwrite(flat, np.zeros((16, 16), 'uint16'))
    assert channels(flat) == []                                # nothing to choose


def test_the_icon_is_a_png_tk_will_accept():
    """Drawn, not shipped, so it has to survive being decoded again."""
    from mitofi.icon import art, png
    data = png(48)
    assert data.startswith(b'\x89PNG\r\n\x1a\n'), data[:8]
    a = art(48)
    assert a.shape == (48, 48, 4) and a.dtype == np.uint8
    assert (a[..., 3] == 0).any() and (a[..., 3] == 255).any()   # a shape, not a square
    assert 0 < a[..., 3].mean() < 255                            # anti-aliased edge
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception:
        return                          # no display; the bytes still checked out
    root.withdraw()
    try:
        p = tk.PhotoImage(data=data)
        assert (p.width(), p.height()) == (48, 48)
    finally:
        root.destroy()


def test_preview_follows_the_run_then_hands_over_on_a_click():
    """The preview must show each image as it is scored, not freeze on the first.

    Tk delivers <<TreeviewSelect>> from the event loop, so a flag raised around
    selection_set is already down when the handler runs: every automatic
    selection looked like a user click and switched following off after image
    one. Following is decided by which row was selected instead.
    """
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception:
        return                          # no display; nothing to drive
    from mitofi.gui import App
    try:
        root.geometry('900x600'); root.update()
        app = App(root)
        img = (np.random.default_rng(0).random((256, 256)) * 500).astype('uint16')
        img[60:200, 100:108] = 4000
        mask = np.zeros((256, 256), bool); mask[60:200, 100:108] = True
        for i, name in enumerate(('a', 'b', 'c'), 1):
            app.add_row(dict(image_id=name, qc_pass=True, index=0.3, band='mixed',
                             confidence=0.6), img, mask, f'{name}.tif', None, i)
            root.update()
        assert app.current == 'c', app.current          # trailed to the last one
        assert app.follow is True

        app.tree.selection_set('a'); root.update()
        assert app.follow is False, 'a click must stop the preview from moving'
        assert app.current == 'a'

        fit = app.photo.width()
        app.set_zoom(4)
        assert app.zoom_lab.get() == '4:1', app.zoom_lab.get()
        assert app.photo.width() > fit, (app.photo.width(), fit)
        app.set_zoom(None)
        assert app.photo.width() == fit
    finally:
        root.destroy()


def test_preview_redraws_at_the_size_of_the_panel():
    """A canvas that grew with the window used to keep painting the old picture."""
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception:
        return
    from mitofi.gui import App
    try:
        root.geometry('900x600'); root.update()
        app = App(root)
        img = (np.random.default_rng(1).random((512, 512)) * 500).astype('uint16')
        mask = np.zeros((512, 512), bool); mask[100:300, 200:210] = True
        app.add_row(dict(image_id='a', qc_pass=True, index=0.3, band='mixed',
                         confidence=0.6), img, mask, 'a.tif', None, 1)
        root.update()
        small = app.photo.width()
        root.geometry('1500x950'); root.update(); app.draw()
        assert app.photo.width() > small, (app.photo.width(), small)
    finally:
        root.destroy()


FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fixtures', 'two_channel_stack.ome.tif')


def test_a_real_acquisition_layout_goes_end_to_end():
    """The README's first example, run: a multi-channel Z-stack in, an index out.

    Every other check here is a unit: this is the only one that reads a file,
    picks a channel, projects, segments and scores. It is also the path that a
    new user hits first, and the one that broke when the index model was a
    pickle -- nothing else in the suite touched the fitted model at all.

    The pixels are real, cropped from the MFN1-null and wild-type MEF fields;
    the stack around them is assembled, because every acquisition in this
    project was projected at ingest and no raw stack survives in the repository.
    """
    from mitofi.io import channels, to_field
    from mitofi.segment import segment
    assert channels(FIXTURE) == ['DAPI 461', 'MitoTracker Deep Red 665']

    img, px, note = to_field(FIXTURE, channel=1)
    assert img.shape == (256, 256), img.shape
    assert abs(px - 0.0691) < 1e-9, px
    assert 'channel 1 of 2' in note and 'max projection over 3 planes' in note, note

    r = score_mask(segment(img), px, img=img)
    assert r['qc_pass'], r['qc_reason']
    assert 0.85 < r['index'] < 1.0, r['index']        # the crop is MFN1-null
    assert r['band'] == 'fragmented', r['band']

    # the channel choice has to matter, or picking one is not a real decision
    other, _, _ = to_field(FIXTURE, channel=0)
    assert abs(float(other.mean()) - float(img.mean())) > 100, (other.mean(), img.mean())


def test_masks_scored_with_their_images_are_gated_on_contrast():
    """Masks alone fall back to the size rule -- which rejects the phenotype.

    Small bright objects are what a fragmented cell looks like. Scored from the
    mask alone there is no way to tell them from noise, so the gate refuses
    them; given the image as well it measures the contrast and keeps them. The
    two runs must disagree, in that direction.
    """
    from mitofi.cli import main
    d = tempfile.mkdtemp()
    imgs, masks = os.path.join(d, 'img'), os.path.join(d, 'mask')
    os.makedirs(imgs); os.makedirs(masks)
    rng = np.random.default_rng(0)
    img = (rng.random((200, 200)) * 60 + 100).astype('uint16')     # dim background
    mask = np.zeros((200, 200), bool)
    for y in range(20, 180, 20):                                    # small, bright puncta
        for x in range(20, 180, 20):
            mask[y:y + 6, x:x + 4] = True
    img[mask] = 3000
    tifffile.imwrite(os.path.join(imgs, 'a.tif'), img)
    tifffile.imwrite(os.path.join(masks, 'a_mask.tif'), mask.astype(np.uint8))

    def run(*extra):
        out = os.path.join(d, f'out{len(extra)}.csv')
        main(['score', '--masks', masks, '--mask-suffix', '_mask.tif',
              '--px-um', '0.0691', '-o', out, *extra])
        import csv as _csv
        with open(out) as fh:
            return next(_csv.DictReader(fh))

    alone = run()
    assert alone['qc_pass'] == 'False', alone
    assert 'no image given' in alone['qc_reason'], alone['qc_reason']

    paired = run(imgs, '--suffix', '.tif')
    assert paired['qc_pass'] == 'True', paired['qc_reason']
    assert paired['index'], paired
    assert paired['local_contrast'], 'contrast was not measured from the image'


def test_a_bad_output_path_fails_before_the_work():
    """An unwritable -o used to raise after every image had been segmented."""
    from mitofi.cli import main
    d = tempfile.mkdtemp()
    tifffile.imwrite(os.path.join(d, 'a.tif'), np.zeros((32, 32), 'uint16'))
    try:
        main(['score', d, '--px-um', '0.0691', '-o', '/proc/nowhere/x.csv'])
    except SystemExit as e:
        assert 'cannot write' in str(e), e
    else:
        raise AssertionError('an unwritable output path must stop the run')

    # a missing but creatable folder is made, not refused
    out = os.path.join(d, 'deeper', 'still', 'x.csv')
    main(['score', d, '--px-um', '0.0691', '-o', out])
    assert os.path.exists(out)


def test_progress_draws_only_for_a_terminal_and_cleans_up_after_itself():
    """The bar goes to stderr and is erased before each row, or the table tears.

    It must also be silent when stderr is redirected: a log file full of redraw
    escapes is worse than no progress at all.
    """
    import io
    from mitofi.cli import Progress, _clock

    class Tty(io.StringIO):
        def isatty(self):
            return True

    quiet = Progress(10, stream=io.StringIO())        # a plain pipe
    quiet.show(3)
    assert quiet.stream.getvalue() == '', 'drew progress into a non-terminal'

    tty = Progress(10, stream=Tty())
    tty.show(0)
    assert 'starting' in tty.stream.getvalue()        # no estimate before the first one
    tty.show(5, 'img_a')
    line = tty.stream.getvalue()
    assert '5/10' in line and 'img_a' in line, line
    assert '\u2588' in line and '\u2591' in line, 'no bar drawn'
    tty.clear()
    assert tty.stream.getvalue().endswith('\r'), 'the line was not wiped'
    assert _clock(90) == '1m30s' and _clock(3700) == '1h01m', _clock(3700)


def test_resume_keeps_finished_rows_and_scores_only_the_rest():
    """An interrupted run must not start over; an hour of segmentation is too
    much to lose to one Ctrl-C."""
    import csv as _csv
    from mitofi.cli import main
    d = tempfile.mkdtemp()
    masks = os.path.join(d, 'mask')
    os.makedirs(masks)
    m = np.zeros((120, 120), bool)
    m[20:100, 50:58] = True                            # one tubule, passes the gate
    for name in ('a', 'b', 'c'):
        tifffile.imwrite(os.path.join(masks, f'{name}_mask.tif'), m.astype(np.uint8))

    out = os.path.join(d, 'r.csv')
    main(['score', '--masks', masks, '--mask-suffix', '_mask.tif',
          '--px-um', '0.0691', '-o', out])
    with open(out) as fh:
        full = list(_csv.DictReader(fh))
    assert len(full) == 3, full

    # throw away the last row, as an interrupt would have
    with open(out, 'w', newline='') as fh:
        w = _csv.DictWriter(fh, fieldnames=full[0].keys())
        w.writeheader(); w.writerows(full[:2])

    main(['score', '--masks', masks, '--mask-suffix', '_mask.tif',
          '--px-um', '0.0691', '-o', out, '--resume'])
    with open(out) as fh:
        again = list(_csv.DictReader(fh))
    assert len(again) == 3, again
    assert [r['image_id'] for r in again] == [r['image_id'] for r in full]
    assert [r['index'] for r in again] == [r['index'] for r in full], 'resume changed a score'


def test_gui_offers_to_resume_a_folder_it_has_already_scored():
    """A stopped window run must not have to start over either.

    Driven without the event loop: what is checked is the wiring -- that a CSV
    left in the folder is noticed, that its rows are carried into the table, and
    that a folder with nothing left to do says so rather than re-segmenting.
    """
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception:
        return
    from mitofi.cli import COLUMNS
    from mitofi.gui import App
    root.withdraw()
    try:
        app = App(root)
        d = tempfile.mkdtemp()
        img = np.zeros((64, 64), 'uint16')
        img[20:44, 30:34] = 3000
        tifffile.imwrite(os.path.join(d, 'a.tif'), img,
                         resolution=(1 / 0.0691, 1 / 0.0691), resolutionunit='CENTIMETER')

        app.folder.set(d); app.scan()
        assert str(app.resume_box['state']) == 'disabled'      # nothing to resume yet
        assert app.resume.get() is False

        import csv as _csv
        with open(os.path.join(d, 'mitofi.csv'), 'w', newline='') as fh:
            w = _csv.DictWriter(fh, fieldnames=COLUMNS, restval='', extrasaction='ignore')
            w.writeheader()
            w.writerow(dict(image_id='a', qc_pass='True', index='0.42', band='mixed',
                            confidence='0.55', px_um='0.0691', n_objects='7'))

        # a CSV with no record of how it was made cannot be carried over: the
        # rows might have come from another channel, scale or model
        app.scan()
        assert str(app.resume_box['state']) == 'disabled', 'resumed an unprovenanced CSV'
        assert 'different run' in app.status.get(), app.status.get()

        from mitofi.cli import _save_manifest
        _save_manifest(os.path.join(d, 'mitofi.csv'), app.run_settings())

        app.scan()
        assert str(app.resume_box['state']) == 'normal', 'an earlier CSV went unnoticed'
        assert app.resume.get() is True
        assert 'already scored' in app.status.get(), app.status.get()

        app.start()
        assert 'nothing left to do' in app.status.get(), app.status.get()
        assert len(app.rows) == 1 and app.rows[0]['index'] == 0.42, app.rows
        assert app.tree.get_children(), 'the carried-over row is not in the table'

        app.resume.set(False)                                   # rescoring is a choice
        app.progress.config(maximum=4)
        app.t0 = None
        assert '/4' in app.eta(1), app.eta(1)
    finally:
        root.destroy()


def test_gui_saves_the_summary_beside_the_per_image_csv():
    """The GUI must write the same two files the CLI does, in the image folder."""
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception:
        return                      # no display; nothing to drive
    root.withdraw()
    from tkinter import filedialog
    from mitofi.gui import App
    try:
        app = App(root)
        d = tempfile.mkdtemp()
        tifffile.imwrite(os.path.join(d, 'x.ome.tif'), np.zeros((8, 8), 'uint16'))
        app.folder.set(d); app.scan()
        assert '1 file' in app.status.get(), app.status.get()

        app.rows = [dict(image_id='a', qc_pass=True, index=0.4, band='mixed',
                         confidence=0.5, px_um=0.069, reduction=''),
                    dict(image_id='b', qc_pass=False, qc_reason='speckle',
                         px_um=0.069, reduction='')]
        filedialog.asksaveasfilename = lambda **k: os.path.join(d, 'mitofi.csv')
        app.save()
        written = sorted(f for f in os.listdir(d) if f.endswith('.csv'))
        assert written == ['mitofi.csv', 'mitofi_summary.csv'], written
        with open(os.path.join(d, 'mitofi_summary.csv')) as fh:
            body = fh.read().splitlines()[1]
        assert body.startswith('2,1,1,0.4'), body     # the rejected row is excluded
    finally:
        root.destroy()


def test_summary_excludes_rejected_rather_than_scoring_them_zero():
    """A rejected image must not drag the mean toward zero. The tool declined to
    score it; averaging in a number it refused to report would invent data."""
    from mitofi.cli import summarise
    rows = [dict(qc_pass=True, index=0.8, band='mixed'),
            dict(qc_pass=True, index=0.6, band='mixed'),
            dict(qc_pass=False, index=None, band='qc_fail', qc_reason='speckle')]
    s = summarise(rows)
    assert s['n_images'] == 3 and s['n_scored'] == 2 and s['n_rejected'] == 1
    assert abs(s['mean_index'] - 0.7) < 1e-9, s
    assert abs(s['median_index'] - 0.7) < 1e-9, s
    assert s['n_mixed'] == 2 and s['n_fragmented'] == 0

    none_ok = summarise([dict(qc_pass=False, index=None, band='qc_fail')])
    assert none_ok['n_scored'] == 0 and 'mean_index' not in none_ok

    one = summarise([dict(qc_pass=True, index=0.5, band='mixed')])
    assert one['sd_index'] == '', one          # sd of a single value is undefined


def test_imagej_tiff_calibration_is_read():
    """A TIFF saved by Fiji is not OME, but carries its scale in resolution tags.
    Missing that would silently force every user to supply --px-um."""
    d = tempfile.mkdtemp()
    a = (np.random.default_rng(0).random((64, 64)) * 300).astype('uint16')

    p = os.path.join(d, 'imagej.tif')
    tifffile.imwrite(p, a, imagej=True, resolution=(1 / 0.0829, 1 / 0.0829),
                     metadata={'unit': 'micron'})
    assert abs(pixel_size_um(p) - 0.0829) < 1e-6, pixel_size_um(p)

    q = os.path.join(d, 'plain.tif')
    tifffile.imwrite(q, a)
    assert pixel_size_um(q) is None, 'uncalibrated file must not invent a scale'

    r = os.path.join(d, 'pixelunit.tif')
    tifffile.imwrite(r, a, imagej=True, resolution=(1.0, 1.0), metadata={'unit': 'pixel'})
    assert pixel_size_um(r) is None, 'a size in pixels is not a physical size'


def test_small_but_crisp_objects_are_not_called_speckle():
    """The defect this gate was rebuilt to fix: MFN1 knockout mitochondria are
    genuinely small, and the old size-only rule threw away three of five images
    that scored 0.93 to 0.98 when allowed through."""
    from mitofi.features import local_contrast
    px = 0.07
    yy, xx = np.mgrid[:400, :400]
    small = np.zeros((400, 400), bool)
    for cy in range(40, 380, 26):
        for cx in range(40, 380, 26):
            small |= np.hypot(yy - cy, xx - cx) < 9      # about 1.3 um across

    rng = np.random.default_rng(0)
    bright = rng.normal(100, 12, small.shape)
    bright[small] += 1500                                # crisp objects, quiet background
    f = features(small, bright, px)
    f['local_contrast'] = local_contrast(bright, small, px)
    assert f['p90_major_um'] < 2.25, f['p90_major_um']   # small enough to trip the old rule
    ok, why = qc(f)
    assert ok, f'crisp small objects rejected: {why}  (contrast {f["local_contrast"]:.1f})'

    faint = rng.normal(100, 12, small.shape)
    faint[small] += 12                                   # same shapes, lost in the noise
    g = features(small, faint, px)
    g['local_contrast'] = local_contrast(faint, small, px)
    bad, why2 = qc(g)
    assert not bad and 'speckle' in why2, (bad, why2)


def test_mask_only_mode_still_falls_back_to_size():
    """With no image there is no contrast to measure, so the size rule applies
    alone and the reason says so, rather than passing something unchecked."""
    rng = np.random.default_rng(0)
    speckle = np.zeros((400, 400), bool)
    for _ in range(400):
        y, x = rng.integers(5, 395, 2)
        speckle[y:y + 4, x:x + 4] = True
    f = features(speckle, None, 0.1)
    ok, why = qc(f)
    assert not ok and 'local contrast could not be checked' in why, why


def test_missing_pixel_size_is_asked_for_but_never_assumed():
    """Prompt a person, refuse a script, and never invent a scale."""
    import sys as _sys
    from mitofi.cli import ask_px

    class FakeStdin:
        def __init__(self, tty): self._tty = tty
        def isatty(self): return self._tty

    real = _sys.stdin
    try:
        _sys.stdin = FakeStdin(True)
        assert ask_px(['a'], 3, ask=lambda _: '0.083') == 0.083
        assert ask_px(['a'], 3, ask=lambda _: '  0.5 ') == 0.5
        assert ask_px(['a'], 3, ask=lambda _: '') is None          # Enter cancels
        for bad in ('0', '-1'):
            try:
                ask_px(['a'], 3, ask=lambda _, b=bad: b)
            except SystemExit:
                pass
            else:
                raise AssertionError(f'{bad} accepted as a pixel size')
        try:
            ask_px(['a'], 3, ask=lambda _: 'wide')
        except SystemExit:
            pass
        else:
            raise AssertionError('non-numeric accepted')

        # a script must never be prompted: it would hang waiting on stdin
        _sys.stdin = FakeStdin(False)
        def must_not_ask(_):
            raise AssertionError('prompted a non-interactive run')
        assert ask_px(['a'], 3, ask=must_not_ask) is None
    finally:
        _sys.stdin = real


def test_table_columns_line_up_and_long_names_are_marked():
    from mitofi.cli import _fit, _row
    assert _fit('short', 10) == 'short'
    long = _fit('a_very_long_image_identifier_indeed', 12)
    assert len(long) == 12 and long.endswith('\u2026'), long   # trimmed, and says so

    w = 14
    header = _row('#', 'image', 'um/px', 'index', 'band', 'conf', 'notes', w)
    body = _row(1, 'img_a', '0.0691', '0.444', 'mixed', '0.51', '', w)
    rejected = _row(2, 'img_b', '0.0691', '\u2014', '\u2014', '\u2014', 'speckle', w)
    # the numeric columns are right-aligned, so their ENDS line up, not starts
    end = lambda line, tok: line.index(tok) + len(tok)
    for line in (body, rejected):
        assert end(line, '0.0691') == end(header, 'um/px'), line
    assert not body.endswith(' '), 'trailing whitespace on an empty notes column'


def test_output_lands_next_to_the_input_not_the_shell_cwd():
    from mitofi.cli import main
    d = tempfile.mkdtemp()
    m = np.zeros((64, 64), bool)
    m[20:44, 30:34] = True                       # one tubule, passes the gate
    tifffile.imwrite(os.path.join(d, 'a.tif'), m.astype(np.uint8))
    here = os.getcwd()
    os.chdir(tempfile.mkdtemp())                 # a different working directory
    try:
        main(['score', '--masks', d, '--px-um', '0.0691'])
        assert os.path.exists(os.path.join(d, 'mitofi.csv')), os.listdir(d)
        assert os.path.exists(os.path.join(d, 'mitofi_summary.csv'))
        assert not os.listdir(os.getcwd()), 'wrote into the shell working directory'
    finally:
        os.chdir(here)


def test_an_impossible_pixel_size_is_refused_not_carried_through():
    """B01. Infinity used to pass the gate and return an index of NaN."""
    from mitofi.features import check_px
    m = np.zeros((64, 64), bool)
    m[30:34, 8:56] = True
    for bad in (0.0, -0.1, float('inf'), float('nan'), None, 'x'):
        try:
            check_px(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted {bad!r} as a pixel size')
        try:
            score_mask(m, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'scored a mask at pixel size {bad!r}')
    r = score_mask(m, 0.1)                 # a real one still works
    assert r['qc_pass'] is False or r['index'] is not None


def test_an_rgb_tiff_is_refused_rather_than_projected_over_y():
    """B02. A 32x40x3 RGB TIFF used to come back as a 40x3 "image"."""
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'rgb.tif')
    tifffile.imwrite(p, np.random.default_rng(0).integers(
        0, 255, (32, 40, 3), dtype=np.uint8), photometric='rgb')
    try:
        a, _, note = to_field(p)
    except ValueError as e:
        assert 'RGB' in str(e), e
    else:
        raise AssertionError(f'reduced an RGB TIFF to {a.shape}: {note}')


def test_anisotropic_and_unknown_calibration_are_refused():
    """B05. X=0.1 Y=0.2 used to return 0.1 and be treated as square."""
    from mitofi.io import CalibrationError
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'aniso.ome.tif')
    tifffile.imwrite(p, np.zeros((16, 16), np.uint16), ome=True,
                     metadata={'axes': 'YX', 'PhysicalSizeX': 0.1,
                               'PhysicalSizeY': 0.2})
    try:
        pixel_size_um(p)
    except CalibrationError as e:
        assert 'not square' in str(e), e
    else:
        raise AssertionError('accepted anisotropic pixels as square')

    q = os.path.join(d, 'square.ome.tif')
    tifffile.imwrite(q, np.zeros((16, 16), np.uint16), ome=True,
                     metadata={'axes': 'YX', 'PhysicalSizeX': 0.1,
                               'PhysicalSizeY': 0.1})
    assert abs(pixel_size_um(q) - 0.1) < 1e-9


def test_bit_depth_is_put_on_the_scale_the_classifier_was_trained_on():
    """An 8-bit field reached the classifier 16x too dim and collapsed.

    And the rule must leave the development images alone: 12-bit data in
    16-bit containers that record no depth.
    """
    from mitofi.io import to_trained_scale
    same = np.array([[0, 4000]], np.uint16)
    out, note = to_trained_scale(same)
    assert out is same and note == '', note                    # Olympus: untouched
    out, note = to_trained_scale(np.array([[0, 255]], np.uint8))
    assert abs(out.max() - 4095) < 1e-3 and '8-bit' in note, note
    out, _ = to_trained_scale(np.array([[0, 255]], np.uint8), significant_bits=8)
    assert abs(out.max() - 4095) < 1e-3
    out, _ = to_trained_scale(np.array([[0, 65535]], np.uint16))
    assert abs(out.max() - 4095) < 1e-3                        # true 16-bit
    out, _ = to_trained_scale(np.array([[0, 9000]], np.uint16), significant_bits=12)
    assert out.max() <= 4095                                   # wrong metadata
    out, _ = to_trained_scale(np.array([[0, 1.0]], np.float32))
    assert abs(out.max() - 4095) < 1e-3                        # float fraction

    # end to end: the same field saved at 8 and at 12 bits gives the same mask
    from mitofi.segment import segment
    img, px, _ = to_field(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'fixtures', 'two_channel_stack.ome.tif'), channel=1)
    img12 = (img.astype(float) / img.max() * 4095).round().astype(np.uint16)
    img8 = (img12 / 4095 * 255).round().astype(np.uint8)
    d = tempfile.mkdtemp()
    tifffile.imwrite(os.path.join(d, 'a8.tif'), img8)
    tifffile.imwrite(os.path.join(d, 'a12.tif'), img12)
    m8 = segment(to_field(os.path.join(d, 'a8.tif'))[0])
    m12 = segment(to_field(os.path.join(d, 'a12.tif'))[0])
    iou = (m8 & m12).sum() / max((m8 | m12).sum(), 1)
    assert m8.any() and iou > 0.9, iou


def _hazy_and_clean_fields():
    """Bright puncta on black, and the same puncta riding on cytoplasmic haze."""
    rng = np.random.default_rng(3)
    clean = np.zeros((256, 256), np.float32)
    for _ in range(120):
        y, x = rng.integers(10, 246, 2)
        clean[y:y + 3, x:x + 3] = 3000
    yy, xx = np.mgrid[:256, :256]
    haze = 2200 * np.exp(-(((yy - 128) ** 2 + (xx - 128) ** 2) / (2 * 70.0 ** 2)))
    return clean, np.clip(clean + haze, 0, 4095).astype(np.float32)


def test_haze_is_measured_and_background_subtraction_removes_it():
    from mitofi.features import haze_fraction
    from mitofi.io import subtract_background
    clean, hazy = _hazy_and_clean_fields()
    mask = clean > 0
    big = np.ones_like(mask)
    h_clean = haze_fraction(clean, mask, 0.07)
    h_hazy = haze_fraction(hazy, big, 0.07)
    assert h_clean < 0.05, h_clean
    assert h_hazy > 0.5, h_hazy
    fixed = subtract_background(hazy, 0.07)
    assert haze_fraction(fixed, mask, 0.07) < 0.2


def test_the_gate_refuses_haze_and_says_how_to_fix_it():
    from mitofi import score as S
    f = dict(n_objects=50, fg_frac=0.05, p90_major_um=3.0, local_contrast=1.1,
             haze=0.9)
    saved = S.MAX_HAZE
    try:
        S.MAX_HAZE = 0.5
        ok, why = S.qc(f)
        assert not ok and why.startswith(S.HAZE_REASON) and 'subtract' in why, why
        f['haze'] = 0.1
        assert S.qc(f)[0]
    finally:
        S.MAX_HAZE = saved


def test_the_channel_can_be_chosen_by_laser_line_when_names_lie():
    """Leica names put "Mito Tracker Red" at 514 nm; the mitochondria were at 594."""
    from mitofi.io import channel_by_excitation, excitations
    from mitofi.gui import _resolve
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'leica.ome.tif')
    tifffile.imwrite(p, np.zeros((3, 16, 16), np.uint8), ome=True, metadata={
        'axes': 'CYX', 'Channel': {'Name': ['Leica/DAPI', 'Leica/Mito Tracker Red',
                                            'Leica/EGFP'],
                                   'ExcitationWavelength': [594.0, 514.0, 488.0]}})
    assert excitations(p) == [594.0, 514.0, 488.0], excitations(p)
    assert channel_by_excitation(p, 594) == 0
    try:
        channel_by_excitation(p, 640)
    except ValueError as e:
        assert '594' in str(e), e                      # says what the file has
    else:
        raise AssertionError('chose a channel for a laser line the file lacks')
    layout, exc = ('Leica/DAPI', 'Leica/Mito Tracker Red', 'Leica/EGFP'), (594.0, 514.0, 488.0)
    assert _resolve(layout, 'excitation 594 nm', exc) == 0
    assert _resolve(layout, 'Leica/Mito Tracker Red  [ex 514 nm]', exc) == 1


def test_the_cytoplasm_floor_is_read_from_each_image():
    """Grainy haze over the cell is cut; the floor follows the image, not a constant."""
    from mitofi.io import subtract_floor
    rng = np.random.default_rng(0)
    for haze in (60.0, 1200.0):                       # dim 8-bit-like and bright 12-bit-like
        img = np.zeros((200, 200), np.float32)
        img[40:160, 40:160] = haze + rng.normal(0, haze * 0.1, (120, 120))   # grainy cytoplasm
        img[60:64, 50:150] = img[100:104, 50:150] = haze * 3                 # two tubes
        out, fl = subtract_floor(img, 0.06)
        assert abs(fl - haze) < 0.1 * haze, (haze, fl)
        assert out[60:64, 50:150].min() > 0.5 * out.max()                   # tubes kept
        assert np.median(out[120:150, 50:150]) < 0.05 * out.max()           # haze cut


def test_fields_coarser_than_training_are_refused():
    """At 0.24 um/px knockouts scored 0.000-0.003 and passed every other check."""
    from mitofi.score import COARSE_REASON, MAX_PX_UM, refusal_kind, score_mask
    m = np.zeros((128, 128), bool); m[20:24, 10:110] = True; m[60:64, 10:110] = True
    s = score_mask(m, 0.24)
    assert not s['qc_pass'] and s['qc_reason'].startswith(COARSE_REASON), s['qc_reason']
    assert refusal_kind(s['qc_reason'])[0] == 'pixels too coarse'
    assert not score_mask(m, MAX_PX_UM)['qc_reason'].startswith(COARSE_REASON)


def test_refusals_are_counted_and_explained_by_reason():
    """A user told only "10 rejected" cannot tell a wrong channel from haze."""
    from mitofi.cli import refusal_lines, summarise
    from mitofi.score import HAZE_REASON, NOISE_REASON
    rows = [dict(image_id='a', qc_pass=False, qc_reason=HAZE_REASON + ': 12%'),
            dict(image_id='b', qc_pass=False, qc_reason=HAZE_REASON + ': 9%'),
            dict(image_id='c', qc_pass=False, qc_reason=NOISE_REASON + ': 1.3'),
            dict(image_id='d', qc_pass=False, qc_reason='Errno 2: no such file'),
            dict(image_id='e', qc_pass=True, qc_reason='', index=0.4, band='mixed')]
    s = summarise(rows)
    assert s['n_rejected'] == 4
    assert s['rejected_diffuse_background'] == 2
    assert s['rejected_noise_texture'] == 1
    assert s['rejected_could_not_be_read'] == 1
    assert s['rejected_speckle'] == 0                    # present even when zero
    lines = refusal_lines(rows)
    assert lines[0].split()[0] == '2' and 'diffuse background' in lines[0]
    assert any('Removing the background' in l for l in lines)


def test_a_run_mixing_rescued_and_untouched_fields_is_flagged():
    """Black-point rows read about 0.2 higher; mixing them with untouched rows
    can turn processing into an apparent difference between conditions."""
    from mitofi.cli import rescue_mix_warning, summarise
    ok = lambda r: dict(qc_pass=True, index=0.4, band='mixed', rescue=r)
    assert rescue_mix_warning([ok(''), ok('')]) == ''
    assert rescue_mix_warning([ok('black point'), ok('black point')]) == ''   # all alike
    w = rescue_mix_warning([ok(''), ok('black point'), ok('background'),
                            dict(qc_pass=False, index=None, rescue='refused')])
    assert '2 of 3' in w and 'black point 1' in w and '0.2 higher' in w, w
    assert 'Black-point' not in rescue_mix_warning([ok(''), ok('background')])
    s = summarise([ok(''), ok('black point'), ok('background')])
    assert s['n_rescued_black_point'] == 1 and s['n_rescued_background'] == 1


def test_large_objects_made_of_noise_are_refused_not_scored():
    """A sponge segmented out of noise texture passed and scored as elongated.

    The speckle rule only examines small objects, so large ones were never
    checked for whether they stand out from their surroundings at all.
    """
    from mitofi import score as S
    f = dict(n_objects=300, fg_frac=0.15, p90_major_um=2.5, local_contrast=1.1,
             haze=0.01, local_snr=S.MIN_LARGE_SNR - 0.1)
    ok, why = S.qc(f)
    assert not ok and why.startswith(S.NOISE_REASON), why
    f['local_snr'] = 3.0                          # real tubules: kept
    assert S.qc(f)[0]
    f.update(p90_major_um=1.5, local_snr=1.3, local_contrast=10.0)
    assert S.qc(f)[0], 'small crisp objects are the speckle rule\'s business'


def test_the_cli_offers_both_rescues_to_refused_fields_and_marks_them():
    import types
    from mitofi import cli, score as S
    rows = [dict(image_id='a', qc_pass=False, px_um=0.07,
                 qc_reason=S.HAZE_REASON + ': 90% ...'),
            dict(image_id='b', qc_pass=True, px_um=0.07, index=0.5, band='mixed',
                 confidence=0.5, qc_reason=''),
            dict(image_id='c', qc_pass=False, px_um=0.07,
                 qc_reason='mask is speckle: ...')]
    seen = []
    def fake_score(iid, path, args, px, paired, background_um=None, black_point=False):
        seen.append((iid, background_um, black_point))
        if iid == 'c' and not black_point:           # background alone does not rescue c
            return dict(image_id=iid, qc_pass=False, px_um=0.07,
                        qc_reason='mask is speckle: ...', rescue='background')
        return dict(image_id=iid, qc_pass=True, px_um=0.07, index=0.9,
                    band='mixed', confidence=0.8, qc_reason='',
                    rescue='black point' if black_point else 'background',
                    reduction='rescued')
    args = types.SimpleNamespace(subtract_background=None,
                                 out=os.path.join(tempfile.mkdtemp(), 'r.csv'))
    real_score, real_stdin = cli._score_one, sys.stdin
    try:
        cli._score_one = fake_score
        sys.stdin = types.SimpleNamespace(isatty=lambda: True)
        cli._offer_background(rows, {'a': 'a.tif', 'b': 'b.tif', 'c': 'c.tif'},
                              args, {}, {}, 10, ask=lambda _: 'y')
    finally:
        cli._score_one, sys.stdin = real_score, real_stdin
    # every refused field is offered background removal; only those still refused
    # get the black point; a passed field is never touched
    assert seen == [('a', cli.BACKGROUND_UM, False), ('c', cli.BACKGROUND_UM, False),
                    ('c', None, True)], seen
    assert rows[0]['rescue'] == 'background' and rows[2]['rescue'] == 'black point'
    assert os.path.exists(args.out)


def test_the_gui_asks_before_subtracting_background_and_asks_once():
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception:
        return                      # no display; nothing to drive
    root.withdraw()
    from mitofi.gui import App
    from mitofi.score import HAZE_REASON
    try:
        app = App(root)
        app.ids = {'a.tif': 'a', 'b.tif': 'b'}
        app.rows = [dict(image_id='a', qc_pass=False, qc_reason=HAZE_REASON + ': x'),
                    dict(image_id='b', qc_pass=True, qc_reason='')]
        launched, asked = [], []
        app.launch = lambda files, background_um=None, black_point=False: \
            launched.append((files, background_um, black_point))

        app.ask_yes = lambda *a: asked.append(1) or False      # user says no
        app.offer_background()
        assert asked and not launched
        app.black_point_pass = False                           # a fresh run

        app.ask_yes = lambda *a: True                          # user says yes
        app.offer_background()
        assert launched == [(['a.tif'], 2.5, False)], launched  # only the refused one

        app.offer_background()                                 # still refused after it
        assert launched[-1] == (['a.tif'], None, True), launched  # black point offered

        asked.clear(); app.ask_yes = lambda *a: asked.append(1) or True
        app.offer_background()                                 # after both re-scores
        assert not asked, 'asked again about the same fields'
    finally:
        root.destroy()


def test_an_uncalibrated_export_can_be_rescued_with_px_um():
    """OME unit "pixel" means no calibration, which --px-um must be able to fix.

    Bio-Formats writes it for any image with no spatial scale. It used to be
    read as an unknown unit and refused, so the error told the user to pass
    --px-um and passing --px-um still exited: a dead end on exactly the files
    the LIF export macro produces when a series has no calibration.
    """
    from mitofi.cli import main
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'uncal.ome.tif')
    img = np.zeros((64, 64), np.uint16)
    img[20:44, 30:34] = 3000
    tifffile.imwrite(p, img, ome=True,
                     metadata={'axes': 'YX', 'PhysicalSizeX': 1.0,
                               'PhysicalSizeXUnit': 'pixel',
                               'PhysicalSizeY': 1.0, 'PhysicalSizeYUnit': 'pixel'})
    assert pixel_size_um(p) is None, pixel_size_um(p)
    out = os.path.join(d, 'r.csv')
    main(['score', d, '--suffix', '.ome.tif', '--px-um', '0.07', '-o', out])
    assert os.path.exists(out)


def test_a_mask_that_is_not_boolean_is_converted_or_refused():
    """B06. 0/255 raised IndexError; 0/1 uint8 made every pixel background."""
    from mitofi.features import as_mask
    m = np.zeros((64, 64), np.uint8)
    m[30:34, 8:56] = 1
    img = np.zeros((64, 64), np.float32)
    img[30:34, 8:56] = 10.0
    a = features(m, img, 0.1)
    b = features((m * 255).astype(np.uint8), img, 0.1)
    c = features(m.astype(bool), img, 0.1)
    assert a['n_objects'] == b['n_objects'] == c['n_objects'] == 1
    assert abs(a['med_area_um2'] - c['med_area_um2']) < 1e-9
    try:
        as_mask(np.arange(64, dtype=np.uint8).reshape(8, 8) % 8)
    except ValueError as e:
        assert 'binary' in str(e) or 'label image' in str(e), e
    else:
        raise AssertionError('a multi-class label image passed as a binary mask')


def test_contrast_and_features_measure_the_same_objects():
    """B10. 4-connected and unfiltered there, 8-connected and filtered here."""
    from mitofi.features import local_contrast, MIN_OBJ_UM2
    from skimage.measure import label as sk_label
    rng = np.random.default_rng(1)
    m = np.zeros((200, 200), bool)
    for _ in range(60):
        y, x = rng.integers(5, 190, 2)
        m[y:y + 5, x:x + 5] = True
    img = rng.normal(10, 1, m.shape).astype(np.float32)
    img[m] += 40
    px = 0.1
    lc = local_contrast(img, m, px)
    assert np.isfinite(lc) and lc > 0, lc
    # the objects it samples are the ones features() reports
    n_feat = features(m, img, px)['n_objects']
    lab = sk_label(m, connectivity=2)
    n_same = sum(1 for k in range(1, lab.max() + 1)
                 if (lab == k).sum() * px * px >= MIN_OBJ_UM2)
    assert n_feat == n_same, (n_feat, n_same)


def test_skeleton_length_is_a_path_not_a_pixel_count():
    """B11. A 40-pixel diagonal spans 5.52 um at 0.1 um/px, not 4.0."""
    from mitofi.features import skeleton_stats
    h = np.zeros((60, 60), bool)
    h[30, 10:50] = True
    d = np.zeros((60, 60), bool)
    for i in range(40):
        d[10 + i, 10 + i] = True
    lh = skeleton_stats(h, 0.1)['skel_len_um']
    ld = skeleton_stats(d, 0.1)['skel_len_um']
    assert abs(lh - 3.9) < 0.05, lh
    assert abs(ld - 5.52) < 0.05, ld
    # a Y meeting in two adjacent pixels is one junction, not two
    y = np.zeros((40, 40), bool)
    y[20, 5:20] = True
    for i in range(10):
        y[20 - i, 20 + i] = True
        y[20 + i, 20 + i] = True
    assert skeleton_stats(y, 0.1)['n_junctions'] == 1


def test_the_two_outputs_may_not_be_the_same_file():
    """B04. -o x.csv --summary-out x.csv overwrote the results with the summary."""
    from mitofi.cli import main
    d = tempfile.mkdtemp()
    tifffile.imwrite(os.path.join(d, 'a.tif'), np.zeros((32, 32), np.uint16))
    same = os.path.join(d, 'x.csv')
    try:
        main(['score', '--masks', d, '--px-um', '0.1', '-o', same,
              '--summary-out', same])
    except SystemExit as e:
        assert 'same file' in str(e), e
    else:
        raise AssertionError('accepted one path for both outputs')


def test_an_interrupted_write_leaves_the_previous_results_intact():
    """B04. The CSV was rewritten in place after every image."""
    from mitofi.cli import _atomic
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'r.csv')
    _atomic(p, lambda fh: fh.write('image_id,index\na,0.5\n'))
    def boom(fh):
        fh.write('image_id,index\n')
        raise KeyboardInterrupt
    try:
        _atomic(p, boom)
    except KeyboardInterrupt:
        pass
    assert open(p).read() == 'image_id,index\na,0.5\n', open(p).read()
    assert not os.path.exists(p + '.part')


def test_resume_refuses_rows_from_a_different_run():
    """B03. Resume trusted the image_id alone, so a new pixel size kept old rows."""
    from mitofi.cli import main
    d = tempfile.mkdtemp()
    m = np.zeros((64, 64), np.uint8)
    m[20:44, 30:34] = 1
    tifffile.imwrite(os.path.join(d, 'a.tif'), m)
    out = os.path.join(d, 'r.csv')
    main(['score', '--masks', d, '--px-um', '0.1', '-o', out])
    first = open(out).read()

    main(['score', '--masks', d, '--px-um', '0.2', '-o', out, '--resume'])
    assert open(out).read() != first, 'a changed pixel size kept the old row'

    os.remove(out + '.run.json')                 # no provenance at all
    try:
        main(['score', '--masks', d, '--px-um', '0.2', '-o', out, '--resume'])
    except SystemExit as e:
        assert 'run.json' in str(e), e
    else:
        raise AssertionError('resumed a CSV with no record of how it was made')


def test_a_fully_resumed_run_still_writes_its_summary():
    """B08. It returned early, leaving the summary missing."""
    from mitofi.cli import main
    d = tempfile.mkdtemp()
    m = np.zeros((64, 64), np.uint8)
    m[20:44, 30:34] = 1
    tifffile.imwrite(os.path.join(d, 'a.tif'), m)
    out = os.path.join(d, 'r.csv')
    spath = os.path.join(d, 'r_summary.csv')
    main(['score', '--masks', d, '--px-um', '0.1', '-o', out])
    os.remove(spath)
    main(['score', '--masks', d, '--px-um', '0.1', '-o', out, '--resume'])
    assert os.path.exists(spath), 'a fully resumed run left no summary'


def test_a_segmented_mask_carries_its_own_pixel_size():
    """B07. The mask branch hardcoded None, so the scale was lost after preflight."""
    from mitofi.cli import main
    d = tempfile.mkdtemp()
    img = np.zeros((64, 64), np.uint16)
    img[20:44, 30:34] = 3000
    tifffile.imwrite(os.path.join(d, 'a.tif'), img, imagej=True,
                     resolution=(1 / 0.0691, 1 / 0.0691), metadata={'unit': 'micron'})
    masks = os.path.join(d, 'masks')
    main(['segment', d, '-o', masks])
    got = pixel_size_um(os.path.join(masks, 'a_mask.tif'))
    assert got and abs(got - 0.0691) < 1e-4, got
    main(['score', '--masks', masks, '-o', os.path.join(d, 'm.csv')])   # no --px-um


def test_identifiers_stay_distinct_when_the_stems_collide():
    """B09. a.tif, a.tiff and a.ome.tif all became `a` in the GUI."""
    from mitofi.gui import _ids
    got = _ids(['a.tif', 'a.tiff', 'a.ome.tif', 'b.tif'])
    assert len(set(got.values())) == 4, got
    assert got['b.tif'] == 'b', got                 # the unambiguous one keeps its stem


def test_compressed_tiffs_are_read():
    """LZW (ImageJ, Zeiss, Fiji exports) and other common TIFF compressions; earlier builds could not."""
    from mitofi.io import to_field
    d = tempfile.mkdtemp()
    a = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 4000)
    for c in ('lzw', 'zlib', 'zstd', 'packbits'):
        p = os.path.join(d, f'{c}.tif')
        tifffile.imwrite(p, a, compression=c)
        assert np.array_equal(np.asarray(to_field(p)[0]), a), c


def test_inputs_skips_directories_and_refuses_an_empty_suffix():
    d = tempfile.mkdtemp()
    os.mkdir(os.path.join(d, 'notanimage.tif'))     # a folder that ends in .tif
    tifffile.imwrite(os.path.join(d, 'real.TIF'), np.zeros((8, 8), np.uint16))
    got = _inputs(d, '.tif')
    assert [i for i, _ in got] == ['real'], got     # case-insensitive, no directory
    try:
        _inputs(d, '')
    except SystemExit:
        pass
    else:
        raise AssertionError('accepted an empty suffix')


def test_a_timepoint_outside_the_file_is_named_not_a_traceback():
    d = tempfile.mkdtemp()
    p = os.path.join(d, 't.ome.tif')
    tifffile.imwrite(p, np.zeros((3, 4, 16, 16), np.uint16),
                     ome=True, metadata={'axes': 'TZYX'})
    try:
        to_field(p, timepoint=9)
    except ValueError as e:
        assert '0..2' in str(e), e
    else:
        raise AssertionError('accepted a timepoint the file does not have')
    a, _, note = to_field(p, timepoint=1)
    assert a.shape == (16, 16) and 'timepoint 1 of 3' in note, note


def test_the_model_is_the_nine_features_it_claims():
    """A refit that silently drops or reorders an input must fail here.

    The order matters as much as the set: predict_proba multiplies the vector by
    the coefficient matrix positionally, so a reordered feature list would score
    every image with the wrong coefficients and raise nothing.
    """
    from mitofi.score import load
    M = load()
    assert M['features'] == [
        'p90_aspect', 'p90_ecc', 'areawt_formfactor', 'med_aspect', 'med_ecc',
        'hole_area_frac', 'areawt_aspect', 'frac_fg_largest',
        'mean_skel_len_per_obj_um'], M['features']
    assert M['classes'] == ['elongated', 'fragmented', 'normal'], M['classes']
    assert np.array(M['coef']).shape == (3, 9), np.array(M['coef']).shape
    assert len(M['mean']) == len(M['scale']) == 9
    # the skeleton feature is the ninth, and it is a path length now
    from mitofi.features import features
    m = np.zeros((200, 200), bool)
    m[100, 20:180] = True
    f = features(m, None, 0.1)
    assert abs(f['mean_skel_len_per_obj_um'] - 15.9) < 0.05, f['mean_skel_len_per_obj_um']


def test_thresholds_are_what_the_paper_says():
    from mitofi.score import MIN_LOCAL_CONTRAST
    assert MIN_P90_MAJOR_UM == 2.25
    assert MIN_LOCAL_CONTRAST == 4.0


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print('ok  ', fn.__name__)
    print(f'\n{len(fns)} checks passed')
