import loci.formats.ImageReader;
import loci.formats.MetadataTools;
import loci.formats.meta.IMetadata;
import loci.common.DebugTools;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Hashtable;

/** One tab-separated line per file read from stdin: acquisition date, sizes,
 *  pixel size, and per channel the Olympus dye name and laser line -- the
 *  fields the ingest macro used to choose the mitochondrial channel. */
public class OirInfo {
    static String g(Hashtable<String, Object> h, String k) {
        Object v = h.get(k);
        return v == null ? "" : v.toString().replace('\t', ' ');
    }
    public static void main(String[] a) throws Exception {
        DebugTools.setRootLevel("OFF");
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, "UTF-8"));
        String path;
        while ((path = in.readLine()) != null) {
            StringBuilder o = new StringBuilder(path);
            try {
                IMetadata m = MetadataTools.createOMEXMLMetadata();
                ImageReader r = new ImageReader();
                r.setMetadataStore(m);
                r.setId(path);
                Hashtable<String, Object> h = r.getGlobalMetadata();
                h.putAll(r.getSeriesMetadata());
                int nc = r.getSizeC();
                o.append('\t').append(m.getImageAcquisitionDate(0) == null ? "" : m.getImageAcquisitionDate(0).getValue());
                o.append('\t').append(r.getSeriesCount()).append('\t').append(r.getSizeX()).append('\t').append(r.getSizeY())
                 .append('\t').append(r.getSizeZ()).append('\t').append(nc).append('\t').append(r.getSizeT());
                o.append('\t').append(m.getPixelsPhysicalSizeX(0) == null ? "" : m.getPixelsPhysicalSizeX(0).value());
                o.append('\t').append(m.getPixelsPhysicalSizeZ(0) == null ? "" : m.getPixelsPhysicalSizeZ(0).value());
                for (int c = 0; c < nc; c++) {
                    String dye = g(h, "channel dyeName #" + (c + 1));
                    Object ex = m.getChannelExcitationWavelength(0, c) == null ? "" : m.getChannelExcitationWavelength(0, c).value();
                    o.append('\t').append(dye).append('|').append(ex);
                }
                r.close();
            } catch (Exception e) {
                o.append("\tERROR ").append(e.toString().replace('\t', ' ').replace('\n', ' '));
            }
            System.out.println(o);
            System.out.flush();
        }
    }
}
