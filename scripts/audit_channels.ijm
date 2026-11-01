// audit_channels.ijm
// Which stack position actually holds the mitochondrial dye, in every .oir?
//
// Olympus labels channels CH1, CH3, CH4 when a detector is unused, so a
// three-channel file can have dye #3 sitting at stack position 2. Any macro that
// matches "channel dyeName #n" and then extracts stack position n takes the
// wrong channel in that case, and the result still looks mitochondrial when the
// other stain is a mitochondrial protein.
//
// This reads only, writes nothing, and produces one row per file:
//   file, sizeC, CH labels, mito dye at stack position, what a naive index gives,
//   and MISMATCH when those differ.
//
// Choose the raw folder, then a folder for the report.

EXT = ".oir";
inDir  = getDirectory("Choose the RAW folder holding the .oir files");
outDir = getDirectory("Choose where to write the report");
csv = outDir + "channel_audit.csv";
File.append("file,sizeC,ch_labels,mito_stack_pos,mito_wavelength,naive_index,mismatch", csv);

paths = collect(inDir, newArray(0));
print("auditing " + paths.length + " files");
setBatchMode(true);
nBad = 0;
for (i = 0; i < paths.length; i++) nBad = nBad + auditOne(paths[i]);
setBatchMode(false);
print("");
print("files where the naive index picks the wrong channel: " + nBad + " of " + paths.length);
print("report: " + csv);
if (nBad > 0)
    print("Those files need re-converting with scripts/oir_to_mip.ijm, which selects " +
          "by wavelength.");


function collect(dir, arr) {
    list = getFileList(dir);
    for (i = 0; i < list.length; i++) {
        p = "" + dir + list[i];
        if (endsWith(list[i], "/")) arr = collect(p, arr);
        else if (endsWith(toLowerCase(list[i]), EXT)) arr = Array.concat(arr, p);
    }
    return arr;
}

function auditOne(path) {
    run("Bio-Formats Importer",
        "open=[" + path + "] color_mode=Default view=Hyperstack stack_order=XYCZT");
    info = "" + getMetadata("Info");
    getDimensions(w, h, nc, nz, nt);
    close("*");

    lines = split(info, "\n");
    labels = chLabels(info);
    waves = channelWaves(info, nc);

    // stack position whose CH label carries a mito dye. The dye lookup is inlined:
    // calling a string-returning helper from inside this function is rejected by
    // the macro interpreter with "Numeric return value expected".
    mitoPos = 0; mitoWave = "NA";
    for (i = 0; i < nc; i++) {
        n = "" + (i + 1);
        if (i < labels.length) n = "" + labels[i];
        isMito = 0;
        want = "channel dyeName #" + n;
        for (j = 0; j < lines.length; j++) {
            if (indexOf("" + lines[j], want) != 0) continue;
            if (indexOf(toLowerCase("" + lines[j]), "mito") >= 0) isMito = 1;
            j = lines.length;
        }
        if (isMito == 1) { mitoPos = i + 1; mitoWave = waves[i]; }
    }
    // what a macro matching dyeName #n and extracting position n would take
    naive = 0;
    for (n = 1; n <= nc; n++) {
        want = "channel dyeName #" + n;
        for (j = 0; j < lines.length; j++) {
            if (indexOf("" + lines[j], want) != 0) continue;
            if (indexOf(toLowerCase("" + lines[j]), "mito") >= 0) naive = n;
            j = lines.length;
        }
    }
    bad = 0;
    if (mitoPos > 0 && naive > 0 && mitoPos != naive) bad = 1;

    labStr = "";
    for (i = 0; i < labels.length; i++) {
        if (i > 0) labStr = "" + labStr + "|";
        labStr = "" + labStr + labels[i];
    }
    File.append("\"" + File.getName(path) + "\"," + nc + ",\"" + labStr + "\"," +
                mitoPos + "," + mitoWave + "," + naive + "," + bad, csv);
    if (bad == 1)
        print("MISMATCH  " + File.getName(path) + "  sizeC=" + nc + "  CH=" + labStr +
              "  mito at stack " + mitoPos + " (" + mitoWave + " nm), naive takes " + naive);
    return bad;
}

function chLabels(info) {
    lines = split(info, "\n");
    out = newArray(0);
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
        if (num != "") out = Array.concat(out, num);
    }
    return out;
}

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
    idxs = newArray(0);
    for (i = 0; i < lines.length; i++) {
        t = indexOf(lines[i], "Channel CH");
        if (t < 0 || indexOf(lines[i], "linked laser index") < 0) continue;
        e = indexOf(lines[i], "=");
        if (e < 0) continue;
        idxs = Array.concat(idxs, parseInt(trim(substring(lines[i], e + 1))));
    }
    // Array.concat, not newArray(n): the latter is numeric and rejects strings
    out = newArray(0);
    for (i = 0; i < nc; i++) {
        v = "NA";
        if (i < idxs.length && idxs[i] >= 0 && idxs[i] < lasers.length)
            v = "" + lasers[idxs[i]];
        out = Array.concat(out, v);
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
