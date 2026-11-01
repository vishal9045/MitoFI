// mito_metadata_extract.ijm  (v5)
// Batch-extracts the metadata the pipeline needs from Olympus .oir files.
// Re-running is safe: files already in the manifest are skipped.
//
// v5 fix: the Info property may be "key<TAB>value", "key = value", or "key: value"
// depending on Bio-Formats version. kv() now tries all three.
// It also writes the raw Info text of the FIRST file to _info_dump.txt next to the
// manifest, so key names can be checked directly if anything still comes back NA.
//
// Usage: paste into the Fiji script editor, set language to "IJ1 Macro", Run.
// Note: expressions starting with a function call are prefixed with "" -
// IJ1 infers expression type from the first term and assumes numeric otherwise.

inputDir = getDirectory("Choose the folder containing .oir files");
outDir   = getDirectory("Choose where to save the manifest CSV");
outPath  = outDir + "metadata_manifest.csv";
dumpPath = outDir + "_info_dump.txt";

EXT = ".oir";               // change for .nd2 / .lif / .czi
DUMPED = false;             // raw Info is written once, for the first file seen

HEADER = "image_id,source_path,session,session_datetime,size_c," +
         "bit_depth,saturation_max,pixel_size_um,optical_res_xy_um," +
         "dyes,pmt_voltages,auto_hv";

// ---------- collect paths already in the manifest ----------
done = newArray(0);
if (File.exists(outPath)) {
    rows = split(File.openAsString(outPath), "\n");
    for (i = 1; i < rows.length; i++) {
        cols = split(rows[i], ",");
        if (cols.length > 1) {
            p = "" + replace(cols[1], fromCharCode(34), "");
            done = Array.concat(done, p);
        }
    }
} else {
    File.append(HEADER, outPath);
}

// ---------- walk ----------
setBatchMode(true);
nDone = walk(inputDir, done, outPath, 0);
setBatchMode(false);
print("Finished. " + nDone + " new file(s) appended to " + outPath);
print("Raw metadata of the first file dumped to " + dumpPath);


// ============================== functions ==============================

function walk(dir, done, outPath, n) {
    list = getFileList(dir);
    for (i = 0; i < list.length; i++) {
        path = "" + dir + list[i];
        if (endsWith(list[i], "/")) {
            n = walk(path, done, outPath, n);
        } else if (endsWith(toLowerCase(list[i]), EXT)) {
            skip = false;
            for (j = 0; j < done.length; j++) {
                if (done[j] == path) skip = true;
            }
            if (skip == false) {
                row = "" + extractRow(path);
                if (lengthOf(row) > 0) {
                    File.append(row, outPath);
                    n = n + 1;
                    print("  + " + list[i]);
                }
            }
        }
    }
    return n;
}

function extractRow(path) {
    run("Bio-Formats Importer",
        "open=[" + path + "] color_mode=Default view=Hyperstack stack_order=XYCZT");
    info = getMetadata("Info");
    close("*");

    // write the raw text once so key names / separators can be inspected
    if (DUMPED == false) {
        File.saveString("" + info, dumpPath);
        DUMPED = true;
    }

    if (lengthOf(info) < 10) {
        print("  !! no metadata read from " + path);
        return "";
    }
    lines = split(info, "\n");

    // session comes from the acquisition timestamp, not the folder name
    created = "" + kv(lines, "general creationDateTime #1");
    session = "NA";
    tpos = indexOf(created, "T");
    if (tpos > 0) session = "" + substring(created, 0, tpos);

    // saturation ceiling: 12-bit data in a 16-bit container saturates at 4095, not 65535
    bits = "" + kv(lines, "BitsPerPixel");
    satMax = "NA";
    if (bits != "NA") satMax = "" + d2s(pow(2, parseFloat(bits)) - 1, 0);

    dyes  = "" + kv(lines, "channel dyeName #1") + ";" + kv(lines, "channel dyeName #2")
          + ";" + kv(lines, "channel dyeName #3") + ";" + kv(lines, "channel dyeName #4");
    volts = "" + kv(lines, "pmt voltage #01") + ";" + kv(lines, "pmt voltage #02")
          + ";" + kv(lines, "pmt voltage #03") + ";" + kv(lines, "pmt voltage #04")
          + ";" + kv(lines, "pmt voltage #05");

    fname   = "" + File.getName(path);
    imageID = fname;
    dot = lastIndexOf(fname, ".");
    if (dot > 0) imageID = "" + substring(fname, 0, dot);

    row = "" + q(imageID) + "," + q(path) + "," + q(session) + "," + q(created) + ","
        + q(kv(lines, "SizeC")) + "," + q(bits) + "," + q(satMax) + ","
        + q(kv(lines, "length x #1")) + "," + q(kv(lines, "opticalResolution x #1")) + ","
        + q(dyes) + "," + q(volts) + "," + q(kv(lines, "pmtCommon autoHV"));

    return row;
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
            if (k == key) {
                v = "" + trim(substring(line, t + sepLen));
                return v;
            }
        }
    }
    return "NA";
}

function trim(s) {
    a = 0;
    b = lengthOf(s);
    while (a < b && (substring(s, a, a + 1) == " " || substring(s, a, a + 1) == "\r")) a = a + 1;
    while (b > a && (substring(s, b - 1, b) == " " || substring(s, b - 1, b) == "\r")) b = b - 1;
    r = "" + substring(s, a, b);
    return r;
}

function q(s) {
    Q = fromCharCode(34);
    t = "" + replace("" + s, Q, "'");
    r = "" + Q + t + Q;
    return r;
}
