GStreamer - ZED Demux
The ZED Demux, zeddemux GStreamer element, allows separation of a single ZED composite stream (RGB Left + RGB Right, or RGB Left + Depth) and creates two separate streams. A third stream is created for metadata if requested.

The zeddemux creates three sink pads:

src_left: left color camera stream
src_aux: right color camera stream or 16-bit depth stream (see is-depth property)
src_left: monocular color camera stream (see is-mono property)
src_data: metadata stream (see stream-data property)
Properties #
  is-depth            : Aux source is GRAY16 depth
                        flags: readable, writable
                        Boolean. Default: false
  is-mono             : Demux is applied to ZED X One monocular stream from zedxonesrc
                        flags: readable, writable
                        Boolean. Default: false
  name                : The name of the object
                        flags: readable, writable, 0x2000
                        String. Default: "zeddemux0"
  parent              : The parent of the object
                        flags: readable, writable, 0x2000
                        Object of type "GstObject"
  stream-data         : Enable binary data streaming on `src_data` pad
                        flags: readable, writable
                        Boolean. Default: false
Example pipelines #
Split the stereo couple and display single streams #
Following a simple pipeline to display on the screen the Left and Right color streams in two different windows:

gst-launch-1.0 zedsrc stream-type=2 ! zeddemux is-depth=false is-mono=false name=demux \
 demux.src_left ! queue ! autovideoconvert ! fpsdisplaysink \
 demux.src_aux ! queue ! autovideoconvert ! fpsdisplaysink


Split the RGB/Depth synchronized stream and display single streams #
The same pipeline of the previous example can be used to display left camera stream and Depth stream in separate windows by simply selecting the stream-type 4 and using the default properties of zeddemux

gst-launch-1.0 zedsrc stream-type=4 ! zeddemux name=demux \
 demux.src_left ! queue ! autovideoconvert ! fpsdisplaysink \
 demux.src_aux ! queue ! autovideoconvert ! fpsdisplaysink


For an example of how to use the stream-data option please refer to the documentation of the zeddatacsvsink element