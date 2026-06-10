# MouseHouse and SqueakView Bill of Materials

This bill of materials lists the components used to build the MouseHouse behavioral platform and SqueakView acquisition system described in the manuscript.

Prices and availability may change over time. Equivalent substitutions may be possible but have not necessarily been validated with the published configuration.

## System scope

The validated build includes:

- a 3D-printed MouseHouse enclosure and integrated camera mount;
- a FLIR Blackfly S monochrome machine-vision camera;
- infrared illumination mounted inside the chamber lid;
- a Jetson Orin Nano Super Developer Kit for edge-compute deployment;
- an RP2040-based controller and local event logger;
- optional capacitive-touch lickometer hardware.

---

## A. Edge-compute hardware

| Component | Part used in validated build | Supplier | Qty per station | Approx. unit price | Notes |
|---|---|---|---:|---:|---|
| Edge computer | NVIDIA Jetson Orin Nano Super Developer Kit | NVIDIA Marketplace / authorized distributor | 1 | $249.00 | Runs JetPack, DeepStream, TensorRT, SqueakView, and the operator GUI |
| Edge-compute storage | User-supplied microSD card or NVMe SSD | User-selected | 1 | User-selected | Select capacity based on operating-system and video-recording needs |
| Ethernet cable | Cat5e or Cat6 cable | Multiple suppliers | 1 | User-selected | Recommended for SSH, file transfer, and remote management |

### Storage note

SqueakView does not require a specific storage brand or model. Users may select a compatible microSD card or NVMe SSD based on recording duration, desired storage capacity, and local purchasing preferences.

---

## B. Camera, optics, and acquisition hardware

| Component | Part used in validated build | Supplier | Qty per station | Approx. unit price | Notes |
|---|---|---|---:|---:|---|
| Machine-vision camera | BFS-U3-16S2M-CS USB 3.1 Blackfly® S Monochrome Camera — Stock #11-507 | Edmund Optics / Teledyne FLIR | 1 | $371.00 | 1.6 MP monochrome global-shutter USB3 camera used with the SqueakView acquisition pipeline |
| USB3 camera cable | Type-A to Micro-B USB 3.1 Locking Cable, 3 m — Stock #86-770 | Edmund Optics / Teledyne FLIR | 1 | $25.00 | Locking USB3 cable used for camera data acquisition |
| Camera GPIO cable | Blackfly® 6-pin GPIO Hirose Connector, 1 m Cable — Stock #88-064 | Edmund Optics | 1 | $38.00 | Used for GPIO, trigger, and synchronization connections |
| C-mount spacer | 5 mm Spacer to Convert CS-Mount Cameras to C-Mount | Edmund Optics | 1 | Confirm current price | Required because the selected camera uses a CS mount and the lens uses a C-mount flange distance |
| Camera lens | 4 mm UC Series Fixed Focal Length Lens | Edmund Optics | 1 | Confirm current price | Fixed focal-length machine-vision lens |
| Lens filter adapter | Filter Adapter M62 × 0.75 from M40 × 0.5 Female | Edmund Optics | 1 | $40.50 | Connects the M62 machine-vision filter to the lens |
| Infrared filter | IR (UV/VIS Cut) M62.0 × 0.75 High Performance Machine Vision Filter — Stock #89-842 | Edmund Optics | 1 | $439.00 | Passes infrared illumination while suppressing visible-light contamination |

---

## C. MouseHouse enclosure, camera mounting, and illumination

The MouseHouse chamber enclosure, lid, and camera mount are 3D printed using the CAD files provided in this repository.

The camera mount is integrated into the printed enclosure design, so no separate commercial camera-mounting hardware is required.

The infrared LED strip is cut to the length required for the chamber geometry and adhered directly to the inside surface of the printed lid using its integrated adhesive backing.

| Component | Part used in validated build | Supplier | Qty per station | Approx. unit price | Notes |
|---|---|---|---:|---:|---|
| MouseHouse enclosure | 3D-printed chamber components from repository CAD files | Lab-fabricated | 1 set | User-selected | Includes chamber body and lid |
| Camera mount | 3D-printed camera mount from repository CAD files | Lab-fabricated | 1 | User-selected | Integrated into the MouseHouse enclosure design |
| 3D-printing material | Filament compatible with repository CAD files | User-selected | As needed | User-selected | Record filament type and print settings used locally |
| Infrared illumination | Infrared 850 nm IR LED Strip Light | Waveform Lighting | Cut to fit chamber lid | $55.00 per 1 m or $195.00 per 5 m | Cut to size and adhered to the inner lid surface |
| LED-strip wiring | Header connection and wiring used in validated build | Lab-assembled | 1 set | User-selected | Connects the LED strip to the controller assembly |
| Assembly hardware | Screws, inserts, and connectors required by repository CAD design | Multiple suppliers | 1 set | User-selected | Add sizes and quantities if not obvious from the CAD files |

---

## D. Controller and sensor-logging hardware

| Component | Part used in validated build | Supplier | Qty per station | Approx. unit price | Notes |
|---|---|---|---:|---:|---|
| Microcontroller and logger | Adafruit Feather RP2040 Adalogger — 8 MB Flash with microSD Card, Product ID 5980 | Adafruit | 1 | $14.95 | Core RP2040 controller with removable-storage logging |
| Capacitive-touch controller | Adafruit 12-Key Capacitive Touch Sensor Breakout — MPR121, Product ID 1982 | Adafruit | 1 | $7.95 | Supports capacitive-touch lick detection |
| I²C cable | STEMMA QT / Qwiic JST SH 4-Pin Cable — 50 mm, Product ID 4399 | Adafruit | 1 | $0.95 | Connects STEMMA QT / Qwiic-compatible modules |
| Real-time clock | DS3231 Precision RTC FeatherWing, Product ID 3028 | Adafruit | 1 | Confirm current price | Provides battery-backed timekeeping |
| RTC backup battery | CR1220 coin cell | Adafruit or equivalent | 1 | User-selected | Required for battery-backed RTC operation |
| Controller microSD card | Compatible microSD card | Multiple suppliers | 1 | User-selected | Stores local controller event logs |
| Feather USB data cable | USB data cable compatible with Feather RP2040 Adalogger | Multiple suppliers | 1 | User-selected | Used for programming, power, and serial communication |
| Controller enclosure | Printed or fabricated electronics enclosure | Lab-fabricated | 1 | User-selected | Protects the controller assembly |

---

## E. Optional lickometer module

The capacitive-touch controller can be used to support lickometer-style behavioral measurements. Physical fluid-delivery components may be selected based on the experimental design.

| Component | Part used in validated build | Supplier | Qty per station | Approx. unit price | Notes |
|---|---|---|---:|---:|---|
| Stainless-steel lick spout | Lab-specific | Multiple suppliers | As needed | User-selected | Connected to an MPR121 touch channel |
| Coaxial cable | Lab-specific | Multiple suppliers | As needed | User-selected | Routes electrode and ground connections |
| Ground connection hardware | Wire and connector hardware | Multiple suppliers | As needed | User-selected | Supports stable capacitive sensing |
| Drinking bottle or reservoir | Lab-specific | Multiple suppliers | As needed | User-selected | Holds fluid |
| Tubing | Lab-specific | Multiple suppliers | As needed | User-selected | Connects reservoir and spout |

---

## F. Software dependencies

The following software is required but does not add direct hardware cost.

| Software | Required for |
|---|---|
| NVIDIA JetPack | Jetson operating system and CUDA stack |
| NVIDIA DeepStream SDK | Real-time video inference |
| Teledyne FLIR Spinnaker SDK | Blackfly S camera access |
| SqueakView repository | Acquisition, recording, operator GUI, and inference |
| MouseHouse controller firmware | RP2040-based controller and sensor logging |
| SqueakPose Studio | Pose-model training and export |

---

## G. Approximate subtotal for known-price core components

The following subtotal includes only components with prices currently specified in this document.

| Component | Approx. unit price |
|---|---:|
| NVIDIA Jetson Orin Nano Super Developer Kit | $249.00 |
| FLIR Blackfly S camera | $371.00 |
| USB3 locking cable | $25.00 |
| Camera GPIO cable | $38.00 |
| Lens filter adapter | $40.50 |
| IR-pass filter | $439.00 |
| Feather RP2040 Adalogger | $14.95 |
| MPR121 capacitive-touch controller | $7.95 |
| STEMMA QT cable | $0.95 |
| 1 m IR LED strip | $55.00 |
| **Known-price subtotal** | **$1,241.35** |

This subtotal does not include the camera lens, C-mount spacer, RTC FeatherWing, storage media, printing materials, wiring, fasteners, controller enclosure, optional lickometer hardware, or local fabrication costs.

---

## H. Substitution and reproducibility notes

- The listed camera, optics, and controller components correspond to the validated MouseHouse and SqueakView build.
- Equivalent substitutions may work but have not necessarily been tested with the published configuration.
- Storage capacity is user-selected.
- The IR LED strip is cut to match the chamber geometry.
- The enclosure and camera mount are fabricated from the repository CAD files.
- Experiment-specific lickometer and fluid-delivery hardware may be adapted to local needs.
