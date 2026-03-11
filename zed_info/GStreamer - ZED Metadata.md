GStreamer - ZED Metadata
The stream of data injected in a GStreamer pipeline by the zedsrc element contains color and depth information plus metadata with sensors and object detection data.

The ZED metadata are defined in the GStreamer library gstzedmeta.

Metadata information #
GstZedSrcMeta: metadata global container
ZedInfo: general information about the ZED camera that acquired the data.
ZedPose: Positional Tracking information.
ZedSensors: data from the camera sensors (if available).
ZedImu: 6 DOF inertial data (if available).
ZedMag: 3 DOF magnetic field data (if available).
ZedEnv: environmental data, atmospheric pressure and internal temperature (if available).
ZedCamTemp: camera CMOS temperatures (if available).
ZedObjectData: object detection data (if available).
Data Structures #
GstZedSrcMeta #
struct _GstZedSrcMeta {
    GstMeta meta;

    ZedInfo info;

    ZedPose pose;
    ZedSensors sens;

    gboolean od_enabled;
    guint8 obj_count;
    guint64 frame_id;
    ZedObjectData objects[256];
};
meta: metadata identifier field.
info: camera information.
pose: camera pose.
sens: sensors data.
od_enabled: indicates if object detection data are available.
obj_count: the number of detected objects [max 256].
frame_id: track the meta/buffer throughout the GStreamer pipeline (when working with source code)
objects: array of detected objects.
ZedInfo #
struct _ZedInfo {
    gint cam_model;
    gint stream_type;
    guint grab_single_frame_width;
    guint grab_single_frame_height;
};
cam_model: camera model (0 : ZED, 1 : ZED Mini, 2 : ZED2).
stream_type: type of stream (0 : Left Image, 1 : Right image, 2 : Stereo couple, 3 : 16 bit depth map, 4 : Left + Depth).
grab_single_frame_width: original width of image data.
grab_single_frame_height: original height of image data.
ZedPose #
struct _ZedPose {
    gboolean pose_avail;
    gint pos_tracking_state;
    gfloat pos[3];
    gfloat orient[3];
};
pose_avail: indicates if camera pose is available.
pos_tracking_state: status of the Positional Tracking algorithm (0 : SEARCHING, 1 : OK, 2 : OFF, 3 : FPS_TOO_LOW)
pos: camera position
orient: camera orientation (Eurel Angles).
ZedSensors #
struct _ZedSensors {
    gboolean sens_avail;
    ZedImu imu;
    ZedMag mag;
    ZedEnv env;
    ZedCamTemp temp;
};
sens_avail: indicates if sensors data are available (only with ZED2 and ZED Mini).
imu: IMU data.
mag: Magnetometer data.
env: environment data.
temp: camera temperatures data
📌 Note: An example about how to retrieve Sensors data is provided with the gstzeddatacsvsink element source code

ZedImu #
struct _ZedImu {
    gboolean imu_avail;
    gfloat acc[3];
    gfloat gyro[3];
    gfloat temp;
};
imu_avail: indicates if IMU data are available (only with ZED2 and ZED Mini).
acc: 3 DOF accelerometer data in [m/s²].
gyro: 3 DOF gyroscope data in [rad/sec].
temp: IMU temperature in [°C].
ZedMag #
struct _ZedMag {
    gboolean mag_avail;
    gfloat mag[3];
};
mag_avail: indicates if magnetometer data are available (only with ZED2).
mag: 3 DOF magnetic field data in [µT].
ZedEnv #
struct _ZedEnv {
    gboolean env_avail;
    gfloat press;
    gfloat temp;
};
env_avail: indicates if environment data are available (only with ZED2).
press: atmospheric pressure in [hPa].
temp: internal camera temperature in [°C].
ZedCamTemp #
struct _ZedCamTemp {
    gboolean temp_avail;
    gfloat temp_cam_left;
    gfloat temp_cam_right;
};
env_avail: indicates if CMOS temperatures data are available (only with ZED2).
temp_cam_left: temperature of the left CMOS sensor in [°C].
temp_cam_right: temperature of the right CMOS sensor in [°C].
ZedObjectData #
struct _ZedObjectData {
    gint id;

    OBJECT_CLASS label;
    OBJECT_SUBCLASS sublabel;
    OBJECT_TRACKING_STATE tracking_state;
    OBJECT_ACTION_STATE action_state;

    gfloat confidence;

    gfloat position[3];
    gfloat position_covariance[6];

    gfloat velocity[3];

    unsigned int bounding_box_2d[4][2];

    /* 3D bounding box of the person represented as eight 3D points
          1 ------ 2
         /        /|
        0 ------ 3 |
        | Object | 6
        |        |/
        4 ------ 7
    */
    gfloat bounding_box_3d[8][3];

    gfloat dimensions[3]; // 3D object dimensions: width, height, length

    gboolean skeletons_avail;

    gfloat keypoint_2d[18][2]; // Negative coordinates -> point not valid
    gfloat keypoint_3d[18][3]; // Nan coordinates -> point not valid

    gfloat head_bounding_box_2d[4][2];
    gfloat head_bounding_box_3d[8][3];
    gfloat head_position[3];
};

```bash
* `id`: univoque identifier of the tracked object.
* `label`: class of the identified object.
* `sublabel`: subclass of the identified object [only for MULTICLASS models]
* `tracking_state`: tracking status of the object.
* `confidence`: confidence level of the detection [0, 100].
* `position`: 3D position of the center of the object.
* `position_covariance`: covariance matrix of the position.
* `velocity`: velocity of the object
* `bounding_box_2d`: 2D image coordinates of the four corners of the bounding box
* `bounding_box_3d`: 3D world coordinates of the eight corners of the bounding box
* `dimensions`: 3D dimensions of the 3D bounding box
* `skeletons_avail`: indicates if a skeleton tracking detection was enabled and if skeleton data are available
* `keypoint_2d`: 2D image coordinates of the eighteen skeleton joints
* `keypoint_3d`: 3D world coordinates of the eighteen skeleton joints
* `head_bounding_box_2d`: 2D image coordinates of the four corners of the bounding box of the head
* `head_bounding_box_3d`: 3D world coordinates of the eight corners of the bounding box of the head
* `head_position`: 3D world coordinates of the position of the center of the head

```C++
enum class OBJECT_CLASS {
    PERSON = 0,
    VEHICLE = 1,
    BAG = 2,
    ANIMAL = 3,
    ELECTRONICS = 4,
    FRUIT_VEGETABLE = 5,
    LAST
};

enum class OBJECT_SUBCLASS {
    PERSON = 0, /**< PERSON */
    BICYCLE = 1, /**< VEHICLE */
    CAR = 2, /**< VEHICLE */
    MOTORBIKE = 3, /**< VEHICLE */
    BUS = 4, /**< VEHICLE */
    TRUCK = 5, /**< VEHICLE */
    BOAT = 6, /**< VEHICLE */
    BACKPACK = 7, /**< BAG */
    HANDBAG = 8, /**< BAG */
    SUITCASE = 9, /**< BAG */
    BIRD = 10, /**< ANIMAL */
    CAT = 11, /**< ANIMAL */
    DOG = 12, /**< ANIMAL */
    HORSE = 13, /**< ANIMAL */
    SHEEP = 14, /**< ANIMAL */
    COW = 15, /**< ANIMAL */
    CELLPHONE = 16, /**< ELECTRONIC */
    LAPTOP = 17, /**< ELECTRONIC */
    BANANA = 18, /**< FRUIT/VEGETABLE */
    APPLE = 19, /**< FRUIT/VEGETABLE */
    ORANGE = 20, /**< FRUIT/VEGETABLE */
    CARROT = 21, /**< FRUIT/VEGETABLE */
    LAST = 22
};

enum class OBJECT_TRACKING_STATE {
    OFF, /**< The tracking is not yet initialized, the object ID is not usable */
    OK, /**< The object is tracked */
    SEARCHING, /**< The object couldn't be detected in the image and is potentially occluded, the trajectory is estimated */
    TERMINATE, /**< This is the last searching state of the track, the track will be deleted in the next retreiveObject */
    LAST
};

enum class OBJECT_ACTION_STATE {
    IDLE = 0, /**< The object is staying static. */
    MOVING = 1, /**< The object is moving. */
    LAST
};
namespace skeleton {

enum class BODY_PARTS {
    NOSE = 0,
    NECK = 1,
    RIGHT_SHOULDER = 2,
    RIGHT_ELBOW= 3,
    RIGHT_WRIST = 4,
    LEFT_SHOULDER = 5,
    LEFT_ELBOW = 6,
    LEFT_WRIST = 7,
    RIGHT_HIP = 8,
    RIGHT_KNEE = 9,
    RIGHT_ANKLE = 10,
    LEFT_HIP = 11,
    LEFT_KNEE = 12,
    LEFT_ANKLE = 13,
    RIGHT_EYE = 14,
    LEFT_EYE = 15,
    RIGHT_EAR = 16,
    LEFT_EAR = 17,
    LAST = 18
};

inline int getIdx(BODY_PARTS part) {
    return static_cast<int>(part);
}

static const std::vector<std::pair< BODY_PARTS, BODY_PARTS>> BODY_BONES{
    {BODY_PARTS::NOSE, BODY_PARTS::NECK},
    {BODY_PARTS::NECK, BODY_PARTS::RIGHT_SHOULDER},
    {BODY_PARTS::RIGHT_SHOULDER, BODY_PARTS::RIGHT_ELBOW},
    {BODY_PARTS::RIGHT_ELBOW, BODY_PARTS::RIGHT_WRIST},
    {BODY_PARTS::NECK, BODY_PARTS::LEFT_SHOULDER},
    {BODY_PARTS::LEFT_SHOULDER, BODY_PARTS::LEFT_ELBOW},
    {BODY_PARTS::LEFT_ELBOW, BODY_PARTS::LEFT_WRIST},
    {BODY_PARTS::RIGHT_SHOULDER, BODY_PARTS::RIGHT_HIP},
    {BODY_PARTS::RIGHT_HIP, BODY_PARTS::RIGHT_KNEE},
    {BODY_PARTS::RIGHT_KNEE, BODY_PARTS::RIGHT_ANKLE},
    {BODY_PARTS::LEFT_SHOULDER, BODY_PARTS::LEFT_HIP},
    {BODY_PARTS::LEFT_HIP, BODY_PARTS::LEFT_KNEE},
    {BODY_PARTS::LEFT_KNEE, BODY_PARTS::LEFT_ANKLE},
    {BODY_PARTS::RIGHT_SHOULDER, BODY_PARTS::LEFT_SHOULDER},
    {BODY_PARTS::RIGHT_HIP, BODY_PARTS::LEFT_HIP},
    {BODY_PARTS::NOSE, BODY_PARTS::RIGHT_EYE},
    {BODY_PARTS::RIGHT_EYE, BODY_PARTS::RIGHT_EAR},
    {BODY_PARTS::NOSE, BODY_PARTS::LEFT_EYE},
    {BODY_PARTS::LEFT_EYE, BODY_PARTS::LEFT_EAR}
};
}
📌 Note: An example about how to use Object Detection data is provided with the gstzedodoverlay element source code