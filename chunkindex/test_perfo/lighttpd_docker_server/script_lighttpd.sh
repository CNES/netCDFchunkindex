#!/bin/bash
# Kill a previous lighttpd server
pkill lighttpd

# Run a HTTP server
lighttpd -Df $1

# Wait one second to let time to the server to start
sleep 3
