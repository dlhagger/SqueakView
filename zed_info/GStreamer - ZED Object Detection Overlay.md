GStreamer - ZED Object Detection Overlay
The ZED Object Detection Overlay, zedodoverlay GStreamer element, is a transform filter that allows drawing object detection bounding boxes on a ZED left color video stream. The incoming data stream must contain ZED Metada/gstreamer, you can use the zeddatamux element to add back metadata to a ZED video stream in case that a GStreamer filter removes them.

OD Multi class overlay road

The source code of this plugin is a complete example of how to process the ZED metadata in a GStreamer element to retrieve object detection data.

Properties #
No properties

Example pipelines #
Display Object Detection bounding boxes #
Following a pipeline to acquire a stream at 720p resolution with RGB and MULTI CLASS Object Detection information and display the results on screen:

gst-launch-1.0 zedsrc stream-type=0 od-enabled=true od-detection-model=0 camera-resolution=3 camera-fps=30 ! queue ! \
zedodoverlay ! queue ! autovideoconvert ! fpsdisplaysink


OD Multi class overlay bridge

Display Skeleton Tracking results #
The same pipeline of the previous example can be used to display Skeleton TRacking results by simply changing the bt-detection-model property of the zedsrc element

gst-launch-1.0 zedsrc stream-type=0 bt-enabled=true bt-detection-model=2 camera-resolution=3 camera-fps=30 ! queue ! \
zedodoverlay ! queue ! autovideoconvert ! fpsdisplaysink


OD Skeleton overlay