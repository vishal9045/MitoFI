import loci.formats.ImageReader;
import loci.formats.MetadataTools;
import loci.formats.meta.IMetadata;
import loci.formats.services.OMEXMLService;
import loci.common.services.ServiceFactory;
import loci.common.DebugTools;

public class OmeDump {
    public static void main(String[] a) throws Exception {
        DebugTools.setRootLevel("ERROR");
        IMetadata meta = MetadataTools.createOMEXMLMetadata();
        ImageReader r = new ImageReader();
        r.setMetadataStore(meta);
        r.setId(a[0]);
        OMEXMLService s = new ServiceFactory().getInstance(OMEXMLService.class);
        System.out.print(s.getOMEXML(meta));
        r.close();
    }
}
