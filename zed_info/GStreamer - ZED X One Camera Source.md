GStreamer - ZED X One Camera Source
The GStreamer element ZED X One Camera Source, zedxonesrc, allows for the injection of the monocular camera color stream from ZED X One to a GStreamer pipeline.

Properties #
  blocksize           : Size in bytes to read per buffer (-1 = default)
                        flags: readable, writable
                        Unsigned Integer. Range: 0 - 4294967295 Default: 4096 
  camera-flip         : Mirror image vertically if camera is flipped
                        flags: readable, writable
                        Boolean. Default: false
  camera-fps          : Camera frame rate
                        flags: readable, writable
                        Enum "GstZedXOneSrcFPS" Default: 30, "30  FPS"
                           (120): 120 FPS          - Only with SVGA. Not with ZED X One 4K
                           (60): 60  FPS          - Not available with 4K mode
                           (30): 30  FPS          - Available with all the resolutions
                           (15): 15  FPS          - Available with all the resolutions
  camera-id           : Select camera from cameraID
                        flags: readable, writable
                        Integer. Range: -1 - 255 Default: -1 
  camera-resolution   : Camera Resolution
                        flags: readable, writable
                        Enum "GstZedXOneSrcResol" Default: 2, "HD1200"
                           (4): 4K               - 3840x2160 (only ZED X One 4K)
                           (3): QHDPLUS          - 3200x1800 (only ZED X One 4K)
                           (2): HD1200           - 1920x1200
                           (1): HD1080           - 1920x1080
                           (0): SVGA             - 960x600 (only ZED X One GS)
  camera-sn           : Select camera from the serial number
                        flags: readable, writable
                        Integer64. Range: 0 - 999999999 Default: 0 
  camera-timeout      : Connection opening timeout in seconds
                        flags: readable, writable
                        Float. Range:             0.5 -           86400 Default:               5 
  ctrl-analog-gain    : Camera control: Analog Gain value
                        flags: readable, writable
                        Integer. Range: 1000 - 30000 Default: 30000 
  ctrl-auto-analog-gain: Enable Automatic Analog Gain
                        flags: readable, writable
                        Boolean. Default: true
  ctrl-auto-analog-gain-range-max: Maximum Analog Gain for the automatic analog gain setting
                        flags: readable, writable
                        Integer. Range: 1000 - 30000 Default: 30000 
  ctrl-auto-analog-gain-range-min: Minimum Analog Gain for the automatic analog gain setting
                        flags: readable, writable
                        Integer. Range: 1000 - 30000 Default: 1000 
  ctrl-auto-digital-gain: Enable Automatic Digital Gain
                        flags: readable, writable
                        Boolean. Default: true
  ctrl-auto-digital-gain-range-max: Maximum Digital Gain for the automatic digital gain setting
                        flags: readable, writable
                        Integer. Range: 1 - 256 Default: 256 
  ctrl-auto-digital-gain-range-min: Minimum Digital Gain for the automatic digital gain setting
                        flags: readable, writable
                        Integer. Range: 1 - 256 Default: 1 
  ctrl-auto-exposure  : Enable Automatic Exposure
                        flags: readable, writable
                        Boolean. Default: true
  ctrl-auto-exposure-range-max: Maximum exposure time in microseconds for the automatic exposure setting
                        flags: readable, writable
                        Integer. Range: 1024 - 66666 Default: 66666 
  ctrl-auto-exposure-range-min: Minimum exposure time in microseconds for the automatic exposure setting
                        flags: readable, writable
                        Integer. Range: 1024 - 66666 Default: 1024 
  ctrl-denoising      : Denoising factor
                        flags: readable, writable
                        Integer. Range: 0 - 100 Default: 50 
  ctrl-digital-gain   : Digital Gain value
                        flags: readable, writable
                        Integer. Range: 1 - 256 Default: 3 
  ctrl-exposure-compensation: Exposure Compensation
                        flags: readable, writable
                        Integer. Range: 0 - 100 Default: 50 
  ctrl-exposure-time  : Exposure time in microseconds
                        flags: readable, writable
                        Integer. Range: 1024 - 66666 Default: 10000 
  ctrl-gamma          : Image gamma
                        flags: readable, writable
                        Integer. Range: 1 - 9 Default: 2 
  ctrl-saturation     : Image saturation
                        flags: readable, writable
                        Integer. Range: 0 - 8 Default: 4 
  ctrl-sharpness      : Image sharpness
                        flags: readable, writable
                        Integer. Range: 0 - 8 Default: 1 
  ctrl-whitebalance-auto: Image automatic white balance
                        flags: readable, writable
                        Boolean. Default: true
  ctrl-whitebalance-temperature: Image white balance temperature
                        flags: readable, writable
                        Integer. Range: 2800 - 6500 Default: 4200 
  do-timestamp        : Apply current stream time to buffers
                        flags: readable, writable
                        Boolean. Default: false
  enable-hdr          : Enable HDR if supported by resolution and frame rate.
                        flags: readable, writable
                        Boolean. Default: false
  name                : The name of the object
                        flags: readable, writable, 0x2000
                        String. Default: "zedxonesrc0"
  num-buffers         : Number of buffers to output before sending EOS (-1 = unlimited)
                        flags: readable, writable
                        Integer. Range: -1 - 2147483647 Default: -1 
  opencv-calibration-file: Optional OpenCV Calibration File
                        flags: readable, writable
                        String. Default: ""
  parent              : The parent of the object
                        flags: readable, writable, 0x2000
                        Object of type "GstObject"
  typefind            : Run typefind before negotiating (deprecated, non-functional)
                        flags: readable, writable, deprecated
                        Boolean. Default: false
  verbose-level       : ZED SDK Verbose level
                        flags: readable, writable
                        Integer. Range: 0 - 999 Default: 1 
Example pipelines #
Display the color stream on the screen with FPS information #
This simple pipeline starts the ZED X One grabbing using the default parameters of the zedxonesrc element and automatically converts (autovideoconvert) the frames to the correct format for the fpsdisplaysink element in order to be displayed on the screen with FPS information.

gst-launch-1.0 zedxonesrc ! queue ! autovideoconvert ! queue ! fpsdisplaysink
The queue element is used to decouple frames in a different queue thread, useful for data synchronization for more complex pipelines.

When using the ZED X One 4K model, you can leverage the full 4K resolution:

gst-launch-1.0 zedxonesrc camera-resolution=3 camera-fps=15 ! queue ! autovideoconvert ! queue ! fpsdisplaysink
When using the ZED X One GS model, you can stream frames at 1200p resolution with 60 Hz freqeuncy:

gst-launch-1.0 zedxonesrc camera-resolution=2 camera-fps=60 ! queue ! autovideoconvert ! queue ! fpsdisplaysink
© 2026 Stereolabs Inc. All Rights Reserved.
