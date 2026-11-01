// ingest_blind_mip.ijm
// Builds a blinded whole-stack-MIP image pool from raw .oir, and writes the
// un-blinding key to a SEPARATE folder. Matches how the first 45 images were made,
// so new files join the existing pool directly.
//
// Depth note: stack depth varies across the archive and mean depth differed across
// arms in the first pool (10.2 to 15.0 planes). Deeper stacks superimpose more and
// read as less fragmented, so depth is a potential confound. Decision 2026-09-04:
// project the whole stack regardless. sizeZ and z_range_um are recorded on every
// row so the effect can be tested, or a fixed-depth pool rebuilt, later.
//
// Blinding: ids are random, files are processed in random order, and NOTHING that
// maps an id to a source file is printed to the Log. That mapping exists only in
// the key folder.
//
// Usage: Fiji script editor, language "IJ1 Macro", Run.
// Run with dry-run ticked first: it resolves every file and writes the key WITHOUT
// producing images, so channel and depth choices can be checked cheaply.

EXT = ".oir";

Dialog.create("Blinded whole-stack MIP ingest");
Dialog.addString("Mito channel dye name contains:", "Mito", 20);
Dialog.addNumber("Fallback mito channel (1-based, 0 = skip file):", 0);
Dialog.addCheckbox("Dry run (write key only, no images)", true);
Dialog.show();
DYE_MATCH = Dialog.getString();
FALLBACK  = Dialog.getNumber();
DRY       = Dialog.getCheckbox();

rawRoot = getDirectory("Choose the RAW root folder (F: drive)");
outDir  = getDirectory("Choose the blinded IMAGES folder (data/images)");
keyDir  = getDirectory("Choose the BLIND KEY folder");

keyPath   = keyDir + "tally_sheet_v2.csv";
labelPath = getDirectory("Choose data/ (for labels_blind.csv)") + "labels_blind.csv";

KEYHDR = "random_id,blind_file,original_file,source_path,arm,session,session_datetime," +
         "sizeZ,planes_used,z_step_um,z_range_um,sizeC,channel_used,dye,px_um," +
         "sat_frac,status";
if (!File.exists(keyPath)) File.append(KEYHDR, keyPath);
if (!File.exists(labelPath)) File.append("image_id,blinded_call,date_rated,notes", labelPath);

// ---------- ids already taken ----------
taken = newArray(0);
ex = getFileList(outDir);
for (i = 0; i < ex.length; i++) {
    nm = ex[i];
    if (startsWith(nm, "img_")) taken = Array.concat(taken, substring(nm, 4, 10));
}

// ---------- already-ingested source paths, so a re-run cannot duplicate ----------
// Full paths are matched as substrings of every csv in the key folder: exact, and
// order-independent of column layout, so it keeps working after the key files are
// merged. The original 45 recorded bare filenames, not paths, so the folders they
// came from are not detectable here - do not re-point the macro at those.
// read EVERY csv in the key folder, so dedupe survives the key files being
// merged, renamed or split later
seen = "";
kf = getFileList(keyDir);
for (i = 0; i < kf.length; i++) {
    if (endsWith(toLowerCase(kf[i]), ".csv"))
        seen = "" + seen + File.openAsString(keyDir + kf[i]);
}

// ---------- collect every source file, then shuffle ----------
paths = newArray(0);
paths = collect(rawRoot, paths);
fresh = newArray(0);
nDup = 0;
for (i = 0; i < paths.length; i++) {
    if (lengthOf(seen) > 0 && indexOf(seen, paths[i]) >= 0) nDup = nDup + 1;
    else fresh = Array.concat(fresh, paths[i]);
}
paths = shuffle(fresh);
print("found " + (paths.length + nDup) + " " + EXT + " files | " + nDup +
      " already in the key, skipped | " + paths.length + " to ingest");

setBatchMode(true);
nOK = 0; nSkip = 0;
for (i = 0; i < paths.length; i++) {
    if (processOne(paths[i]) == 1) nOK = nOK + 1; else nSkip = nSkip + 1;
}
setBatchMode(false);

print("ingested " + nOK + " | skipped " + nSkip);
print("key: " + keyPath);
if (DRY) print("DRY RUN - no images written. Untick dry run to produce them.");
print("No id-to-source mapping was printed. That lives only in the key folder.");


// ============================== functions ==============================

function collect(dir, arr) {
    list = getFileList(dir);
    for (i = 0; i < list.length; i++) {
        p = "" + dir + list[i];
        if (endsWith(list[i], "/")) arr = collect(p, arr);
        else if (endsWith(toLowerCase(list[i]), EXT)) arr = Array.concat(arr, p);
    }
    return arr;
}

function shuffle(a) {
    for (i = a.length - 1; i > 0; i--) {
        j = floor(random * (i + 1));
        t = a[i]; a[i] = a[j]; a[j] = t;
    }
    return a;
}

function newId() {
    hex = "0123456789abcdef";
    ok = false;
    while (ok == false) {
        s = "";
        for (k = 0; k < 6; k++) {
            r = floor(random * 16);
            s = "" + s + substring(hex, r, r + 1);
        }
        ok = true;
        for (k = 0; k < taken.length; k++) if (taken[k] == s) ok = false;
    }
    taken = Array.concat(taken, s);
    return s;
}

function processOne(path) {
    run("Bio-Formats Importer",
        "open=[" + path + "] color_mode=Default view=Hyperstack stack_order=XYCZT");
    info = "" + getMetadata("Info");
    getDimensions(w, h, nc, nz, nt);
    getPixelSize(unit, pw, ph);

    lines = split(info, "\n");
    created = "" + kv(lines, "general creationDateTime #1");
    session = "NA";
    tp = indexOf(created, "T");
    if (tp > 0) session = "" + substring(created, 0, tp);
    zstep = "" + kv(lines, "distance z #1");

    // resolve the mitochondrial channel by dye name, fall back to a fixed index
    ch = 0; dye = "NA";
    for (c = 1; c <= nc; c++) {
        d = "" + kv(lines, "channel dyeName #" + c);
        if (d != "NA" && indexOf(toLowerCase(d), toLowerCase(DYE_MATCH)) >= 0) {
            ch = c; dye = d;
        }
    }
    if (ch == 0 && FALLBACK > 0 && FALLBACK <= nc) { ch = FALLBACK; dye = "fallback_index"; }
    if (ch == 0) {
        close("*");
        writeKey("", path, session, created, nz, 0, zstep, "NA", nc, 0, "NA", pw, "NA",
                 "skip_channel_unresolved");
        return 0;
    }

    // whole-stack max projection of the mitochondrial channel
    run("Duplicate...", "title=crop duplicate channels=" + ch + "-" + ch +
                        " slices=1-" + nz);
    run("Z Project...", "projection=[Max Intensity]");
    proj = getTitle();
    selectWindow(proj);

    getStatistics(area, mean, mn, mx, sd);
    nBins = 65536;
    getHistogram(vals, cnts, nBins);
    satMax = 4095;                       // 12-bit data in a 16-bit container
    satPix = 0;
    for (b = satMax; b < nBins; b++) satPix = satPix + cnts[b];
    satFrac = satPix / (w * h);

    id = "" + newId();
    blindName = "img_" + id + ".ome.tif";
    if (DRY == false) saveAs("Tiff", outDir + blindName);

    zrange = "NA";
    if (zstep != "NA") zrange = "" + d2s(parseFloat(zstep) * nz, 3);

    writeKey(id, path, session, created, nz, nz, zstep, zrange, nc, ch, dye, pw,
             d2s(satFrac, 5), "ok");
    if (DRY == false) File.append("img_" + id + ",,,", labelPath);

    close("*");
    return 1;
}

function writeKey(id, path, session, created, nz, used, zstep, zrange, nc, ch, dye,
                  pw, satFrac, status) {
    arm = "" + File.getName(File.getParent(path));
    blindName = "";
    if (lengthOf(id) > 0) blindName = "img_" + id + ".ome.tif";
    row = "" + q(id) + "," + q(blindName) + "," + q(File.getName(path)) + "," + q(path)
        + "," + q(arm) + "," + q(session) + "," + q(created) + "," + q("" + nz)
        + "," + q("" + used) + "," + q("" + zstep) + "," + q("" + zrange)
        + "," + q("" + nc) + "," + q("" + ch) + "," + q(dye) + "," + q("" + pw)
        + "," + q("" + satFrac) + "," + q(status);
    File.append(row, keyPath);
}

// look up a key in the Info lines, tolerating tab / " = " / ": " separators
function kv(lines, key) {
    for (i = 0; i < lines.length; i++) {
        line = "" + lines[i];
        t = indexOf(line, "\t");  sepLen = 1;
        if (t < 1) { t = indexOf(line, " = "); sepLen = 3; }
        if (t < 1) { t = indexOf(line, ": ");  sepLen = 2; }
        if (t > 0) {
            k = "" + trim(substring(line, 0, t));
            if (startsWith(k, "- ")) k = "" + trim(substring(k, 2));
            if (k == key) return "" + trim(substring(line, t + sepLen));
        }
    }
    return "NA";
}

function trim(s) {
    a = 0; b = lengthOf(s);
    while (a < b && (substring(s, a, a+1) == " " || substring(s, a, a+1) == "\r")) a = a + 1;
    while (b > a && (substring(s, b-1, b) == " " || substring(s, b-1, b) == "\r")) b = b - 1;
    return "" + substring(s, a, b);
}

function q(s) {
    Q = fromCharCode(34);
    return "" + Q + replace("" + s, Q, "'") + Q;
}
