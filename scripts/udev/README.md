# PlatformIO USB rules

`99-platformio-udev.rules` is an unmodified upstream snapshot downloaded on
2026-11-09 from:

https://raw.githubusercontent.com/platformio/platformio-core/develop/platformio/assets/system/99-platformio-udev.rules

Installation guidance:
https://docs.platformio.org/en/latest/core/installation/udev-rules.html

Copyright PlatformIO; distributed under Apache License 2.0, included in
`LICENSE.platformio`. Preserve the upstream header when updating this snapshot.

The upstream RP2040 rule covers the bootloader vendor `2e8a`. Normal Feather
serial access remains covered by the setup script's `dialout` membership.
