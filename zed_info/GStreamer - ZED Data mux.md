GStreamer - ZED Data mux
The ZED Data Mux, zeddatamux GStreamer element, allows the addition of ZED Metadata to a ZED video stream. This is useful if the metadata are removed by a transform filter element that does not automatically propagate metadata.

Properties #
No properties

Example pipelines #
Inject ZED metadata information in a stream #
Example pipeline to resize 2K RGB and depth stream to VGA resolution for display purposes remuxing the Object Detection metadata as input for the zedodoverlay element:

gst-launch-1.0 zeddatamux name=mux \
 zedsrc stream-type=4 camera-resolution=0 camera-fps=15 od-detection-model=0 od-enabled=true ! \
 zeddemux stream-data=true is-depth=true name=demux \
 demux.src_aux ! queue ! autovideoconvert ! videoscale ! video/x-raw,width=672,height=376 ! queue ! fpsdisplaysink \
 demux.src_data ! mux.sink_data \
 demux.src_left ! queue ! videoscale ! video/x-raw,width=672,height=376 ! mux.sink_video \
 mux.src ! queue ! zedodoverlay ! queue ! \
 autovideoconvert ! fpsdisplaysink


Line 1: a zeddatamux element is defined with name mux
Line 2: the zedsrc element is configured to acquire Left and Depth data with HD2K resolution at 30 FPS enabling Object Detection with MULTI-CLASS model.
Line 3: a zeddemux element is defined to split the pipeline into 3 separate branches, one to elaborate the Left Color stream, one to elaborate the Depth stream and one with binary data created from Sensors and Detected Objects metadata that are injected in the mux element.
Line 4: the depth stream is processed by a videoscale transform filter element to reduce the size of the image and is finally displayed on the screen with FPS info.
Line 5: the left stream is processed by a videoscale transform filter element to reduce the size of the image and injected in the mux element
Line 6: the muxed stream from the mux element is processed by the zedodoverlay transform filter element to draw object detection data.
Line 7: object detection result is displayed on screen with FPS info.
The zeddatamux element is important to be sure that the stream filtered by zedodoverlay contains the Object Detection metadata that could be removed from the original ZED stream by the videoscale transform filter in case it does not handle this kind of metadata.

📌 Note: The ZED metadata are designed in order to handle image stream transformation in scale; in this particular case, the object bounding boxes are correctly rendered even if the filtered images have a size different from the grabbed frames.