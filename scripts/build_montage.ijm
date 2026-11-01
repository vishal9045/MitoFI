// build_montage.ijm
// Tiles every image in a folder onto one canvas at NATIVE scale, so a single Labkit
// labeling can train across brightness, pixel size and frame size at once.
//
// Nothing is resampled: pixel sizes differ between tiles on purpose, because that is
// what teaches the classifier both scales. Tiles are placed on a grid sized to the
// largest frame, so mixed 512/1024 images sit side by side with black padding.
//
// Writes montage.tif plus montage_tiles.csv (which tile is which image).
//
// Usage: Fiji script editor, language "IJ1 Macro", Run.

srcDir = getDirectory("Choose the folder of images to tile (data/montage_src)");
outDir = getDirectory("Choose where to save the montage");

list = getFileList(srcDir);
imgs = newArray(0);
for (i = 0; i < list.length; i++)
    if (endsWith(toLowerCase(list[i]), ".tif") || endsWith(toLowerCase(list[i]), ".tiff"))
        imgs = Array.concat(imgs, list[i]);
if (imgs.length == 0) exit("no images found");

// pass 1: largest frame decides the grid cell
setBatchMode(true);
cw = 0; ch = 0;
for (i = 0; i < imgs.length; i++) {
    open(srcDir + imgs[i]);
    getDimensions(w, h, c, s, f);
    if (w > cw) cw = w;
    if (h > ch) ch = h;
    close();
}
cols = floor(sqrt(imgs.length));
if (cols * cols < imgs.length) cols = cols + 1;
rows = floor(imgs.length / cols);
if (rows * cols < imgs.length) rows = rows + 1;

newImage("montage", "16-bit black", cols * cw, rows * ch, 1);
csv = outDir + "montage_tiles.csv";
File.saveString("tile_index,col,row,x,y,width,height,image_file\n", csv);

for (i = 0; i < imgs.length; i++) {
    col = i % cols;
    row = floor(i / cols);
    x = col * cw;
    y = row * ch;

    open(srcDir + imgs[i]);
    src = getTitle();
    getDimensions(w, h, c, s, f);
    run("Select All");
    run("Copy");
    close();

    selectWindow("montage");
    makeRectangle(x, y, w, h);
    run("Paste");

    File.append(i + "," + col + "," + row + "," + x + "," + y + "," + w + "," + h
                + "," + imgs[i], csv);
}

selectWindow("montage");
run("Select None");
saveAs("Tiff", outDir + "montage.tif");
setBatchMode(false);

print("montage: " + cols + " x " + rows + " grid, cell " + cw + "x" + ch
      + ", canvas " + (cols*cw) + "x" + (rows*ch));
print("tiles listed in " + csv);
print("Annotate montage.tif in Labkit with two classes, then save the classifier.");
