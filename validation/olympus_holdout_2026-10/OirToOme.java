import loci.formats.ImageReader;
import loci.formats.MetadataTools;
import loci.formats.meta.IMetadata;
import loci.formats.out.OMETiffWriter;
import loci.common.DebugTools;
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;

/** Convert each "input<TAB>output" line on stdin from any Bio-Formats format
 *  (here Olympus .oir) to OME-TIFF: every plane, channel and Z, with the
 *  original OME metadata (pixel size, laser lines, acquisition date). Series 0
 *  only. Prints "ok <output>" or "ERROR <input> <reason>" per line. */
public class OirToOme {
    public static void main(String[] a) throws Exception {
        DebugTools.setRootLevel("OFF");
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, "UTF-8"));
        String line;
        while ((line = in.readLine()) != null) {
            String[] p = line.split("\t");
            File out = new File(p[1]);
            try {
                if (out.exists()) { System.out.println("ok " + p[1]); continue; }
                out.getParentFile().mkdirs();
                File tmp = new File(p[1] + ".part.ome.tif");
                tmp.delete();
                IMetadata m = MetadataTools.createOMEXMLMetadata();
                ImageReader r = new ImageReader();
                r.setMetadataStore(m);
                r.setId(p[0]);
                r.setSeries(0);
                OMETiffWriter w = new OMETiffWriter();
                w.setMetadataRetrieve(m);
                w.setBigTiff(true);
                w.setId(tmp.getPath());
                w.setSeries(0);
                for (int i = 0; i < r.getImageCount(); i++) w.saveBytes(i, r.openBytes(i));
                w.close();
                r.close();
                if (!tmp.renameTo(out)) throw new Exception("rename failed");
                System.out.println("ok " + p[1]);
            } catch (Exception e) {
                System.out.println("ERROR " + p[0] + " " + e.toString().replace('\n', ' '));
            }
            System.out.flush();
        }
    }
}
