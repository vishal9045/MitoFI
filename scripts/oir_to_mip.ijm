// oir_to_mip.ijm
// Convert raw Olympus .oir into whole-stack maximum-intensity projections that
// scripts/../mitofi (the mitofi tool) can score.
//
// Choose an input folder and an output folder. Nothing else is asked.
//
// This is the plain converter. It does no blinding, writes no key and keeps the
// original file names - use ingest_blind_mip.ijm if you are building a pool for
// rating, where the blinding matters.
//
// For each .oir found, including in subfolders:
//   - resolve the mitochondrial channel by dye name
//   - maximum-project the whole stack, matching how the pilot images were made
//   - save a calibrated TIFF, so pixel size travels with the image
//   - append the pixel size to px_um.csv in the output folder, as a fallback
//
// Then:
//   mitofi score <output folder> --suffix .tif
//
// Usage: Fiji, script editor, language "IJ1 Macro", Run.

EXT = ".oir";

inDir  = getDirectory("Choose the folder holding the .oir files");
outDir = getDirectory("Choose the output folder for the projections");

// ---------- probe the first file and ask which channel to keep ----------
// Selection is by excitation wavelength, not by position in the stack. Channel
// order is not stable across acquisitions, and Olympus labels channels CH1, CH3,
// CH4 when a detector is unused, so the label and the stack position disagree.
// Matching on the laser resolves both.
probePaths = collect(inDir, newArray(0));
if (probePaths.length == 0) exit("No " + EXT + " files under " + inDir);

setBatchMode(true);
run("Bio-Formats Importer",
    "open=[" + probePaths[0] + "] color_mode=Default view=Hyperstack stack_order=XYCZT");
probeInfo = "" + getMetadata("Info");
getDimensions(pw_, ph_, probeC, probeZ, probeT);
close("*");
setBatchMode(false);

waves = channelWaves(probeInfo, probeC);      // excitation nm per stack position
dyes  = channelDyes(probeInfo, probeC);       // dye name per stack position

// built with Array.concat rather than newArray(n): newArray(n) makes a NUMERIC
// array, and writing a string into one of its elements makes the interpreter
// demand a numeric value and fail with "Numeric return value expected"
choices = newArray(0);
for (i = 0; i < probeC; i++) {
    lab = "stack " + (i + 1);
    if (waves[i] != "NA") lab = "" + lab + "  -  " + waves[i] + " nm";
    if (dyes[i]  != "NA") lab = "" + lab + "  -  " + dyes[i];
    choices = Array.concat(choices, lab);
}
def = 0;
for (i = 0; i < probeC; i++) if (indexOf(toLowerCase(dyes[i]), "mito") >= 0) def = i;

Dialog.create("Which channel holds the mitochondria?");
Dialog.addMessage("Read from " + File.getName(probePaths[0]) + "  (" + probeC + " channels)");
Dialog.addChoice("Channel:", choices, choices[def]);
Dialog.addCheckbox("Match the same wavelength in every file (recommended)", true);
Dialog.show();
picked   = Dialog.getChoice();
BY_WAVE  = Dialog.getCheckbox();

PICK_IDX = 1;
for (i = 0; i < probeC; i++) if (choices[i] == picked) PICK_IDX = i + 1;
PICK_WAVE = waves[PICK_IDX - 1];
PICK_DYE  = dyes[PICK_IDX - 1];
if (BY_WAVE && PICK_WAVE == "NA") {
    BY_WAVE = false;
    print("no wavelength recorded for that channel - falling back to dye name");
}
print("selected: " + picked);
if (BY_WAVE) print("matching every file on " + PICK_WAVE + " nm");
else print("using stack position " + PICK_IDX + " in every file");

csvPath = outDir + "px_um.csv";
if (!File.exists(csvPath)) File.append("image_id,px_um", csvPath);

paths = collect(inDir, newArray(0));
print("found " + paths.length + " " + EXT + " files under " + inDir);
if (paths.length == 0) print("nothing to do - check the folder and that files end in " + EXT);

setBatchMode(true);
nOK = 0; nSkip = 0;
for (i = 0; i < paths.length; i++) {
    if (processOne(paths[i]) == 1) nOK = nOK + 1; else nSkip = nSkip + 1;
}
setBatchMode(false);

print("converted " + nOK + " | skipped " + nSkip);
print("projections: " + outDir);
print("pixel sizes: " + csvPath);
print("");
print("score them with:");
print("  mitofi score \"" + outDir + "\" --suffix .tif");
print("if the pixel size does not come through, add:  --px-um-csv \"" + csvPath + "\"");


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

// strip the extension and anything awkward in a file name, so the output name is
// safe on every platform and usable as an id in the CSV
function cleanName(path) {
    n = "" + File.getName(path);
    d = lastIndexOf(n, ".");
    if (d > 0) n = substring(n, 0, d);
    out = "";
    for (i = 0; i < lengthOf(n); i++) {
        c = substring(n, i, i + 1);
        if (indexOf("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_", c) >= 0)
            out = "" + out + c;
        else out = "" + out + "_";
    }
    return out;
}

function processOne(path) {
    run("Bio-Formats Importer",
        "open=[" + path + "] color_mode=Default view=Hyperstack stack_order=XYCZT");
    info = "" + getMetadata("Info");
    getDimensions(w, h, nc, nz, nt);
    getPixelSize(unit, pw, ph);

    // find the requested channel in THIS file, by wavelength where possible
    w = channelWaves(info, nc);
    dl = channelDyes(info, nc);
    ch = 0; dye = "NA";
    if (BY_WAVE) {
        for (c = 1; c <= nc; c++) if (w[c - 1] == PICK_WAVE) { ch = c; dye = dl[c - 1]; }
        if (ch == 0) {                       // wavelength absent: try the dye name
            for (c = 1; c <= nc; c++)
                if (PICK_DYE != "NA" && dl[c - 1] == PICK_DYE) { ch = c; dye = dl[c - 1]; }
        }
    } else if (PICK_IDX <= nc) {
        ch = PICK_IDX; dye = dl[PICK_IDX - 1];
    }
    if (ch == 0 && nc == 1) { ch = 1; dye = dl[0]; }
    if (ch == 0) {
        print("SKIP  " + File.getName(path) + ": no channel at " + PICK_WAVE +
              " nm (" + nc + " channels here). Scoring another stain would give a " +
              "confident number for the wrong organelle, so this file is left out.");
        close("*");
        return 0;
    }

    run("Duplicate...", "title=crop duplicate channels=" + ch + "-" + ch +
                        " slices=1-" + nz);
    run("Z Project...", "projection=[Max Intensity]");
    selectWindow(getTitle());

    // carry the calibration onto the projection so the saved TIFF holds it
    if (pw > 0) setVoxelSize(pw, ph, 1, unit);

    name = "" + cleanName(path);
    saveAs("Tiff", outDir + name + ".tif");
    File.append(name + "," + d2s(pw, 6), csvPath);
    wl = "" + w[ch - 1];
    print("ok    " + name + "  stack " + ch + "  " + wl + " nm  " + dye + "  " +
          nz + " planes  " + d2s(pw, 5) + " " + unit + "/px");

    close("*");
    return 1;
}

// ---------- channel to excitation wavelength ----------
// Olympus records "Laser LD488 ID = ..." for each installed line, and
// "Channel CH3 linked laser index = 4" pointing into that list in the order the
// lasers appear. Channel labels can skip (CH1, CH3, CH4 in a three-channel file),
// so the label is matched to its position in the stack by order of appearance.
function laserList(info) {
    out = newArray(0);
    lines = split(info, "\n");
    for (i = 0; i < lines.length; i++) {
        t = indexOf(lines[i], "Laser LD");
        if (t < 0) continue;
        rest = substring(lines[i], t + 8);
        num = "";
        for (k = 0; k < lengthOf(rest); k++) {
            c = substring(rest, k, k + 1);
            if (indexOf("0123456789", c) < 0) k = lengthOf(rest);
            else num = "" + num + c;
        }
        if (num == "") continue;
        dup = false;
        for (k = 0; k < out.length; k++) if (out[k] == num) dup = true;
        if (!dup) out = Array.concat(out, num);
    }
    return out;
}

function channelWaves(info, nc) {
    lasers = laserList(info);
    lines = split(info, "\n");
    labels = newArray(0);            // CH numbers, in the order they appear
    idxs = newArray(0);              // the laser index each is linked to
    for (i = 0; i < lines.length; i++) {
        t = indexOf(lines[i], "Channel CH");
        if (t < 0 || indexOf(lines[i], "linked laser index") < 0) continue;
        rest = substring(lines[i], t + 10);
        num = "";
        for (k = 0; k < lengthOf(rest); k++) {
            c = substring(rest, k, k + 1);
            if (indexOf("0123456789", c) < 0) k = lengthOf(rest);
            else num = "" + num + c;
        }
        e = indexOf(lines[i], "=");
        if (num == "" || e < 0) continue;
        labels = Array.concat(labels, num);
        idxs = Array.concat(idxs, parseInt(trim(substring(lines[i], e + 1))));
    }
    out = newArray(0);                       // string array, see the note above
    for (i = 0; i < nc; i++) {
        v = "NA";
        if (i < labels.length && idxs[i] >= 0 && idxs[i] < lasers.length)
            v = "" + lasers[idxs[i]];
        out = Array.concat(out, v);
    }
    return out;
}

// dye names, indexed by the CH label so a skipped channel does not shift them.
//
// The lookup is inlined rather than calling kv(): the ImageJ macro interpreter
// rejects this nested user-function call with "Numeric return value expected",
// and no amount of string-forcing at the call site avoids it. Plain indexOf and
// substring here work everywhere.
function channelDyes(info, nc) {
    lines = split(info, "\n");
    labels = newArray(0);
    for (i = 0; i < lines.length; i++) {
        t = indexOf(lines[i], "Channel CH");
        if (t < 0 || indexOf(lines[i], "linked laser index") < 0) continue;
        rest = substring(lines[i], t + 10);
        num = "";
        for (k = 0; k < lengthOf(rest); k++) {
            c = substring(rest, k, k + 1);
            if (indexOf("0123456789", c) < 0) k = lengthOf(rest);
            else num = "" + num + c;
        }
        if (num != "") labels = Array.concat(labels, num);
    }

    out = newArray(0);
    for (i = 0; i < nc; i++) {
        n = "" + (i + 1);
        if (i < labels.length) n = "" + labels[i];      // use the CH label if known
        want = "channel dyeName #" + n;
        d = "NA";
        for (j = 0; j < lines.length; j++) {
            line = "" + lines[j];
            if (indexOf(line, want) != 0) continue;     // must start with the key
            e = indexOf(line, "=");
            if (e < 0) continue;
            v = substring(line, e + 1);
            while (lengthOf(v) > 0 && startsWith(v, " ")) v = substring(v, 1);
            while (lengthOf(v) > 0 && (endsWith(v, " ") || endsWith(v, "\r")))
                v = substring(v, 0, lengthOf(v) - 1);
            if (lengthOf(v) > 0) d = "" + v;
            j = lines.length;
        }
        out = Array.concat(out, d);
    }
    return out;
}

function trim(s) {
    while (lengthOf(s) > 0 && (startsWith(s, " ") || startsWith(s, "\t")))
        s = substring(s, 1);
    while (lengthOf(s) > 0 && (endsWith(s, " ") || endsWith(s, "\t") || endsWith(s, "\r")))
        s = substring(s, 0, lengthOf(s) - 1);
    return s;
}
