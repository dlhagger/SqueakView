GStreamer - ZED RTSP Server
The Real Time Streaming Protocol (RTSP) is a network control protocol designed for use in entertainment and communications systems to control streaming media servers. The protocol is used for establishing and controlling media sessions between endpoints.

ZED RTSP Server is a GStreamer application for the Linux operating system that allows instantiating an RTSP server from a text launch pipeline using the “gst-launch” style.

The ZED RTSP server listens for connection requests on a TCP/IP port and allows to delivery of several ZED media streams simultaneously and independently of each other.

Usage #
The RTSP Server can be started

 gst-zed-rtsp-launch [OPTION?] PIPELINE-DESCRIPTION
Help Options:

-h, --help -> Show help options.
--help-all -> Show all help options.
--help-gst -> Show GStreamer Options.
Application Options:

-p, --port=PORT -> Port to listen on (default: 8554).
-a, --address=HOST -> Host address (default: 127.0.0.1).
Example #
Acquire a ZED stream using the default configuration, and provide RTSP connection streams using H264 encoding:

gst-zed-rtsp-launch zedsrc ! videoconvert ! 'video/x-raw, format=(string)I420' ! x264enc ! rtph264pay pt=96 name=pay0
A console log provides information about the RTSP server address at which an RTSP-capable client can connect to acquire the ZED stream:

 ZED RTSP Server 
-----------------
 * Stream ready at rtsp://127.0.0.1:8554/zed-stream
📌 Note: it is mandatory to define at least one payload named pay0; it is possible to define multiple payloads using an increasing index (i.e. pay1, pay2, …).

A client can be created using GStreamer with the standard gst-launch application:

gst-launch-1.0 rtspsrc location=rtsp://${SERVER_IP}:8554/zed-stream latency=0 ! decodebin ! fpsdisplaysink
The decodebin element automatically creates the required pipeline to decode the incoming stream and convert it for the fpsdisplaysink sink element.