GStreamer - ZED Camera Source
The ZED Camera Source, zedsrc GStreamer element, is the main plugin of the package, it allows injecting the ZED data in a GStreamer pipeline and getting the information provided by the ZED SDK.

Properties #
  area-file-path      : Area localization file that describes the surroundings, saved from a previous tracking session.
                        flags: readable, writable
                        String. Default: ""
  blocksize           : Size in bytes to read per buffer (-1 = default)
                        flags: readable, writable
                        Unsigned Integer. Range: 0 - 4294967295 Default: 4096 
  bt-allow-red-prec   : Set to TRUE to enable Body Tracking reduced inference precision 
                        flags: readable, writable
                        Boolean. Default: false
  bt-body-fitting     : Set to TRUE to enable Body Tracking model fitting 
                        flags: readable, writable
                        Boolean. Default: true
  bt-body-tracking    : Set to TRUE to enable body tracking across images flow 
                        flags: readable, writable
                        Boolean. Default: true
  bt-confidence       : Minimum Detection Confidence
                        flags: readable, writable
                        Float. Range:               0 -             100 Default:              20 
  bt-detection-model  : Body Tracking Model
                        flags: readable, writable
                        Enum "GstZedSrcBtModel" Default: 1, "Body Tracking MEDIUM"
                           (0): Body Tracking FAST - Keypoints based, specific to human skeleton, real time performance even on Jetson or low end GPU cards
                           (1): Body Tracking MEDIUM - Keypoints based, specific to human skeleton, compromise between accuracy and speed
                           (2): Body Tracking ACCURATE - Keypoints based, specific to human skeleton, state of the art accuracy, requires powerful GPU
  bt-enabled          : Set to TRUE to enable Body Tracking
                        flags: readable, writable
                        Boolean. Default: false
  bt-format           : Body Tracking format
                        flags: readable, writable
                        Enum "GstZedSrcBtFormat" Default: 1, "Body 34 Key Points"
                           (0): Body 18 Key Points - 18 keypoints format. Basic Body format
                           (1): Body 34 Key Points - 34 keypoints format. Body format, requires body fitting enabled
                           (2): Body 38 Key Points - 38 keypoints format. Body format, including feet simplified face and hands
  bt-max-range        : Maximum Detection Range
                        flags: readable, writable
                        Float. Range:              -1 -           20000 Default:           20000 
  bt-min-keypoints    : Specify the Minimum keypoints threshold.
                        flags: readable, writable
                        Integer. Range: 0 - 70 Default: 5 
  bt-prediction-timeout-s: Body Tracking prediction timeout (sec)
                        flags: readable, writable
                        Float. Range:               0 -               1 Default:             0.2 
  bt-smoothing        : Smoothing of the fitted fused skeleton
                        flags: readable, writable
                        Float. Range:               0 -               1 Default:               0 
  camera-disable-self-calib: Disable the self calibration processing when the camera is opened
                        flags: readable, writable
                        Boolean. Default: false
  camera-fps          : Camera frame rate
                        flags: readable, writable
                        Enum "GstZedSrcFPS" Default: 15, "15  FPS"
                           (120): 120 FPS          - only SVGA (GMSL2) resolution
                           (100): 100 FPS          - only VGA (USB3) resolution
                           (60): 60  FPS          - VGA (USB3), HD720, HD1080 (GMSL2), and HD1200 (GMSL2) resolutions
                           (30): 30  FPS          - VGA (USB3), HD720 (USB3) and HD1080 (USB3/GMSL2) resolutions
                           (15): 15  FPS          - all resolutions (NO GMSL2)
  camera-id           : Select camera from cameraID
                        flags: readable, writable
                        Integer. Range: 0 - 255 Default: 0 
  camera-image-flip   : Use the camera in forced flip/no flip or automatic mode
                        flags: readable, writable
                        Enum "GstZedSrcFlip" Default: 2, "Auto"
                           (0): No Flip          - Force no flip
                           (1): Flip             - Force flip
                           (2): Auto             - Auto mode (ZED2/ZED2i/ZED-M only)
  camera-resolution   : Camera Resolution
                        flags: readable, writable
                        Enum "GstZedSrcRes" Default: 6, "Default value for the camera model"
                           (0): HD2K (USB3)      - 2208x1242
                           (1): HD1080 (USB3/GMSL2) - 1920x1080
                           (2): HD1200 (GMSL2)   - 1920x1200
                           (3): HD720 (USB3)     - 1280x720
                           (4): SVGA (GMSL2)     - 960x600
                           (5): VGA (USB3)       - 672x376
                           (6): Default value for the camera model - Automatic
  camera-sn           : Select camera from camera serial number
                        flags: readable, writable
                        Integer64. Range: 0 - 9223372036854775807 Default: 0 
  confidence-threshold: Specify the Depth Confidence Threshold
                        flags: readable, writable
                        Integer. Range: 0 - 100 Default: 50 
  coordinate-system   : 3D Coordinate System
                        flags: readable, writable
                        Enum "GstZedsrcStreamType" Default: 0, "Image"
                           (0): Image            - Standard coordinates system in computer vision. Used in OpenCV.
                           (1): Left handed, Y up - Left-Handed with Y up and Z forward. Used in Unity with DirectX.
                           (2): Right handed, Y up - Right-Handed with Y pointing up and Z backward. Used in OpenGL.
                           (3): Right handed, Z up - Right-Handed with Z pointing up and Y forward. Used in 3DSMax.
                           (4): Left handed, Z up - Left-Handed with Z axis pointing up and X forward. Used in Unreal Engine.
                           (5): Right handed, Z up, X fwd - Right-Handed with Z pointing up and X forward. Used in ROS (REP 103).
  ctrl-aec-agc        : Camera automatic gain and exposure
                        flags: readable, writable
                        Boolean. Default: true
  ctrl-aec-agc-roi-h  : Auto gain/exposure ROI height (-1 to not set ROI)
                        flags: readable, writable
                        Integer. Range: -1 - 1242 Default: -1 
  ctrl-aec-agc-roi-side: Auto gain/exposure ROI side
                        flags: readable, writable
                        Enum "GstZedsrcSide" Default: 2, "BOTH"
                           (0): LEFT             - Left side only
                           (1): RIGHT            - Right side only
                           (2): BOTH             - Left and Right side
  ctrl-aec-agc-roi-w  : Auto gain/exposure ROI width (-1 to not set ROI)
                        flags: readable, writable
                        Integer. Range: -1 - 2208 Default: -1 
  ctrl-aec-agc-roi-x  : Auto gain/exposure ROI top left 'X' coordinate (-1 to not set ROI)
                        flags: readable, writable
                        Integer. Range: -1 - 2208 Default: -1 
  ctrl-aec-agc-roi-y  : Auto gain/exposure ROI top left 'Y' coordinate (-1 to not set ROI)
                        flags: readable, writable
                        Integer. Range: -1 - 1242 Default: -1 
  ctrl-brightness     : Image brightness
                        flags: readable, writable
                        Integer. Range: 0 - 8 Default: 4 
  ctrl-contrast       : Image contrast
                        flags: readable, writable
                        Integer. Range: 0 - 8 Default: 4 
  ctrl-exposure       : Camera exposure
                        flags: readable, writable
                        Integer. Range: 0 - 100 Default: 80 
  ctrl-exposure-range-max: Maximum exposure time in microseconds for the automatic exposure setting
                        flags: readable, writable
                        Integer. Range: 28 - 66000 Default: 66000 
  ctrl-exposure-range-min: Minimum exposure time in microseconds for the automatic exposure setting
                        flags: readable, writable
                        Integer. Range: 28 - 66000 Default: 28 
  ctrl-gain           : Camera gain
                        flags: readable, writable
                        Integer. Range: 0 - 100 Default: 60 
  ctrl-gamma          : Image gamma
                        flags: readable, writable
                        Integer. Range: 1 - 9 Default: 8 
  ctrl-hue            : Image hue
                        flags: readable, writable
                        Integer. Range: 0 - 11 Default: 0 
  ctrl-led-status     : Camera LED on/off
                        flags: readable, writable
                        Boolean. Default: true
  ctrl-saturation     : Image saturation
                        flags: readable, writable
                        Integer. Range: 0 - 8 Default: 4 
  ctrl-sharpness      : Image sharpness
                        flags: readable, writable
                        Integer. Range: 0 - 8 Default: 4 
  ctrl-whitebalance-auto: Image automatic white balance
                        flags: readable, writable
                        Boolean. Default: true
  ctrl-whitebalance-temperature: Image white balance temperature
                        flags: readable, writable
                        Integer. Range: 2800 - 6500 Default: 4600 
  depth-maximum-distance: Maximum depth value
                        flags: readable, writable
                        Float. Range:             500 -           40000 Default:           20000 
  depth-minimum-distance: Minimum depth value
                        flags: readable, writable
                        Float. Range:             100 -            3000 Default:             300 
  depth-mode          : Depth Mode
                        flags: readable, writable
                        Enum "GstZedsrcDepthMode" Default: 0, "NONE"
                           (6): NEURAL_PLUS      - More accurate Neural disparity estimation, Requires AI module.
                           (5): NEURAL           - End to End Neural disparity estimation, requires AI module
                           (3): ULTRA            - Computation mode favorising edges and sharpness. Requires more GPU memory and computation power.
                           (2): QUALITY          - Computation mode designed for challenging areas with untextured surfaces.
                           (1): PERFORMANCE      - Computation mode optimized for speed.
                           (0): NONE             - This mode does not compute any depth map. Only rectified stereo images will be available.
  depth-stabilization : Enable depth stabilization
                        flags: readable, writable
                        Integer. Range: 0 - 100 Default: 1 
  do-timestamp        : Apply current stream time to buffers
                        flags: readable, writable
                        Boolean. Default: false
  enable-area-memory  : This mode enables the camera to remember its surroundings. This helps correct positional tracking drift, and can be helpful for positioning different cameras relative to one other in space.
                        flags: readable, writable
                        Boolean. Default: true
  enable-imu-fusion   : This setting allows you to enable or disable IMU fusion. When set to false, only the optical odometry will be used.
                        flags: readable, writable
                        Boolean. Default: true
  enable-pose-smoothing: This mode enables smooth pose correction for small drift correction.
                        flags: readable, writable
                        Boolean. Default: true
  enable-positional-tracking: Enable positional tracking
                        flags: readable, writable
                        Boolean. Default: false
  fill-mode           : Specify the Depth Fill Mode
                        flags: readable, writable
                        Boolean. Default: false
  initial-world-transform-pitch: Pitch orientation of the camera in the world frame when the camera is started
                        flags: readable, writable
                        Float. Range:               0 -             360 Default:               0 
  initial-world-transform-roll: Roll orientation of the camera in the world frame when the camera is started
                        flags: readable, writable
                        Float. Range:               0 -             360 Default:               0 
  initial-world-transform-x: X position of the camera in the world frame when the camera is started
                        flags: readable, writable
                        Float. Range:   -3.402823e+38 -    3.402823e+38 Default:               0 
  initial-world-transform-y: Y position of the camera in the world frame when the camera is started
                        flags: readable, writable
                        Float. Range:   -3.402823e+38 -    3.402823e+38 Default:               0 
  initial-world-transform-yaw: Yaw orientation of the camera in the world frame when the camera is started
                        flags: readable, writable
                        Float. Range:               0 -             360 Default:               0 
  initial-world-transform-z: Z position of the camera in the world frame when the camera is started
                        flags: readable, writable
                        Float. Range:   -3.402823e+38 -    3.402823e+38 Default:               0 
  input-stream-ip     : Specify IP adress when using streaming input
                        flags: readable, writable
                        String. Default: ""
  input-stream-port   : Specify port when using streaming input
                        flags: readable, writable
                        Integer. Range: 1 - 65535 Default: 30000 
  measure3D-reference-frame: Specify the 3D Reference Frame
                        flags: readable, writable
                        Enum "GstZedsrc3dMeasRefFrame" Default: 0, "WORLD"
                           (0): WORLD            - The positional tracking pose transform will contains the motion with reference to the world frame.
                           (1): CAMERA           - The  pose transform will contains the motion with reference to the previous camera frame.
  name                : The name of the object
                        flags: readable, writable, 0x2000
                        String. Default: "zedsrc0"
  num-buffers         : Number of buffers to output before sending EOS (-1 = unlimited)
                        flags: readable, writable
                        Integer. Range: -1 - 2147483647 Default: -1 
  od-allow-reduced-precision-inference: Set to TRUE to allow inference to run at a lower precision to improve runtime
                        flags: readable, writable
                        Boolean. Default: false
  od-conf-animal      : Animal Detection Confidence Threshold
                        flags: readable, writable
                        Float. Range:              -1 -             100 Default:              35 
  od-conf-bag         : Bag Detection Confidence Threshold
                        flags: readable, writable
                        Float. Range:              -1 -             100 Default:              35 
  od-conf-electronics : Electronics Detection Confidence Threshold
                        flags: readable, writable
                        Float. Range:              -1 -             100 Default:              35 
  od-conf-fruit-vegetables: Fruit/Vegetables Detection Confidence Threshold
                        flags: readable, writable
                        Float. Range:              -1 -             100 Default:              35 
  od-conf-people      : People Detection Confidence Threshold
                        flags: readable, writable
                        Float. Range:              -1 -             100 Default:              35 
  od-conf-sport       : Sport Detection Confidence Threshold
                        flags: readable, writable
                        Float. Range:              -1 -             100 Default:              35 
  od-conf-vehicle     : Vehicle Detection Confidence Threshold
                        flags: readable, writable
                        Float. Range:              -1 -             100 Default:              35 
  od-confidence       : Minimum Detection Confidence
                        flags: readable, writable
                        Float. Range:               0 -             100 Default:              50 
  od-detection-filter-mode: Object Detection Filter Mode
                        flags: readable, writable
                        Enum "GstZedSrcOdFilterMode" Default: 2, "(null)"
                           (1): (null)           - SDK will not apply any preprocessing to the detected objects
                           (1): (null)           - SDK will remove objects that are in the same 3D position as an already tracked object (independant of class ID)
                           (2): (null)           - SDK will remove objects that are in the same 3D position as an already tracked object of the same class ID.
  od-detection-model  : Object Detection Model
                        flags: readable, writable
                        Enum "GstZedSrcOdModel" Default: 1, "Object Detection Multi class MEDIUM"
                           (0): Object Detection Multi class FAST - Any objects, bounding box based
                           (1): Object Detection Multi class MEDIUM - Any objects, bounding box based, compromise between accuracy and speed
                           (2): Object Detection Multi class ACCURATE - Any objects, bounding box based, more accurate but slower than the base model
                           (3): Person Head FAST - Bounding Box detector specialized in person heads, particularly well suited for crowded environments, the person localization is also improved
                           (4): Person Head ACCURATE - Bounding Box detector specialized in person heads, particularly well suited for crowded environments, the person localization is also improved, more accurate but slower than the base model
  od-enable-tracking  : Set to TRUE to enable tracking for the detected objects
                        flags: readable, writable
                        Boolean. Default: true
  od-enabled          : Set to TRUE to enable Object Detection
                        flags: readable, writable
                        Boolean. Default: false
  od-max-range        : Maximum Detection Range
                        flags: readable, writable
                        Float. Range:              -1 -           20000 Default:           20000 
  od-prediction-timeout-s: Object prediction timeout (sec)
                        flags: readable, writable
                        Float. Range:               0 -               1 Default:             0.2 
  opencv-calibration-file: Optional OpenCV Calibration File
                        flags: readable, writable
                        String. Default: ""
  parent              : The parent of the object
                        flags: readable, writable, 0x2000
                        Object of type "GstObject"
  pos-depth-min-range : This setting allows you to change the minmum depth used by the SDK for Positional Tracking.
                        flags: readable, writable
                        Float. Range:              -1 -           65535 Default:              -1 
  positional-tracking-mode: Positional tracking mode
                        flags: readable, writable
                        Enum "GstZedsrcPtMode" Default: 2, "GEN_3"
                           (0): GEN_1            - Generation 1
                           (1): GEN_2            - Generation 2
                           (1): GEN_3            - Generation 2
  roi                 : Enable region of interest filtering
                        flags: readable, writable
                        Boolean. Default: false
  roi-h               : Region of interest height (-1 to not set ROI)
                        flags: readable, writable
                        Integer. Range: -1 - 1242 Default: -1 
  roi-w               : Region of intererst width (-1 to not set ROI)
                        flags: readable, writable
                        Integer. Range: -1 - 2208 Default: -1 
  roi-x               : Region of interest top left 'X' coordinate (-1 to not set ROI)
                        flags: readable, writable
                        Integer. Range: -1 - 2208 Default: -1 
  roi-y               : Region of interest top left 'Y' coordinate (-1 to not set ROI)
                        flags: readable, writable
                        Integer. Range: -1 - 1242 Default: -1 
  sdk-verbose         : ZED SDK Verbose level
                        flags: readable, writable
                        Integer. Range: 0 - 1000 Default: 0 
  set-as-static       : Set to TRUE if the camera is static
                        flags: readable, writable
                        Boolean. Default: false
  set-floor-as-origin : This mode initializes the tracking to be aligned with the floor plane to better position the camera in space.
                        flags: readable, writable
                        Boolean. Default: false
  set-gravity-as-origin: This setting allows you to override of 2 of the 3 rotations from initial-world-transform using the IMU gravity default: true
                        flags: readable, writable
                        Boolean. Default: true
  stream-type         : Image stream type
                        flags: readable, writable
                        Enum "GstZedSrcCoordSys" Default: 0, "Left image [BGRA]"
                           (0): Left image [BGRA] - 8 bits- 4 channels Left image
                           (1): Right image [BGRA] - 8 bits- 4 channels Right image
                           (2): Stereo couple up/down [BGRA] - 8 bits- 4 channels bit Left and Right
                           (3): Depth image [GRAY16_LE] - 16 bits depth
                           (4): Left and Depth up/down [BGRA] - 8 bits- 4 channels Left and Depth(image)
  svo-file-path       : Input from SVO file
                        flags: readable, writable
                        String. Default: ""
  texture-confidence-threshold: Specify the Texture Confidence Threshold
                        flags: readable, writable
                        Integer. Range: 0 - 100 Default: 100 
  typefind            : Run typefind before negotiating (deprecated, non-functional)
                        flags: readable, writable, deprecated
                        Boolean. Default: false
ZED Video Demuxer Element properties #
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
Display the left image on the screen #
This simple pipeline starts the ZED grabbing using the default parameters of the zedsrc element and automatically converts (autovideoconvert) the frames to the correct format for the fpsdisplaysink element in order to be displayed on the screen with FPS information.

gst-launch-1.0 zedsrc ! autovideoconvert ! queue ! fpsdisplaysink
The queue element is used to decouple frames in a different queue thread, useful for data synchronization for more complex pipelines.



Display color stereo couple with VGA resolution at 100 FPS #
This pipeline is the same as the previous example, but the properties of zedsrc are changed to set VGA resolution and 100 FPS retrieving the synchronized stereo image.

gst-launch-1.0 zedsrc camera-resolution=5 camera-fps=100 stream-type=2 ! autovideoconvert ! queue ! fpsdisplaysink


Display depth map with HD720 resolution at 60 FPS #
Again the same pipeline, but the parameters are changed to retrieve only the not normalized depth map with HD720 resolution at 60 FPS.

gst-launch-1.0 zedsrc camera-resolution=3 camera-fps=60 stream-type=3 ! autovideoconvert ! queue ! fpsdisplaysink


H264 streaming over UDP on the local network #
GStreamer is famous for its capability of providing an easy way of transmitting media data over networks. TCP, UDP, RTP and many other protocols are available, together with many encoding algorithms like H264, H265, VP8, VP9, matroska, ecc.

The following gst-launch pipeline acquires an RGB stream from a ZED camera, displays it on the local machine and sends it over the network using the UDP protocol with H264 encoding:

gst-launch-1.0 zedsrc ! timeoverlay ! tee name=split has-chain=true ! \
 queue ! autovideoconvert ! fpsdisplaysink \
 split. ! queue max-size-time=0 max-size-bytes=0 max-size-buffers=0 ! autovideoconvert ! \
 x264enc byte-stream=true tune=zerolatency speed-preset=ultrafast bitrate=3000 ! \
 h264parse ! rtph264pay config-interval=-1 pt=96 ! queue ! \
 udpsink clients=192.168.1.169:5000 max-bitrate=3000000 sync=false async=false


Line 1: the zedsrc element is configured to acquire the left camera RGB data with a resolution of 1280×720 (HD720) at 30 FPS. The timestamp is printed on the image using the timeoverlay element and the pipeline is splitted into two branches using tee.
Line 2: the first branch of the pipeline is used to display the frames on the local machine with FPS and timestamp information.
Line 3: the second branch of the pipeline is used to encode the stream using the H264 encoder (x264enc) configured to reduce the latency to the minimum possible.
Line 4: the H264 stream is parsed (h264parse) and a RTP payload is created (rtph264pay).
Line 5: the RTP payload is sent using UDP (udpsink) to the client at address 192.168.1.169 (change the address according to your listener IP) which will be listening on port 5000.
📌 Note: the element queue max-size-time=0 max-size-bytes=0 max-size-buffers=0 is necessary to synchronize the two branches generated by the tee element, otherwise the stream gets frozen after a few milliseconds.

The receiver machine can acquire the stream using this simple pipeline:

gst-launch-1.0 udpsrc port=5000 ! application/x-rtp,clock-rate=90000,payload=96 ! \
 queue ! rtph264depay ! h264parse ! avdec_h264 ! \
 queue ! autovideoconvert ! fpsdisplaysink


Line 1: an UDP source (udpsrc) is configured to listen on port 5000 for a RTP payload of type 96.
Line 2: the payload is elaborated (rtph264depay) to extract the H264 data to be parsed (h264parse) and then decoded (avdec_h264).
Line 3: the decoded stream is finally converted and displayed on the screen with FPS information.
Comparing visually the timestamp printed on each frame on the sender and on the receiver machines, it is possible to grossly evaluate the latency of the network.