FROM ubuntu:22.04

COPY 80proxy /etc/apt/apt.conf.d/80proxy
RUN apt-get update
RUN apt-get install -y lighttpd
RUN mkdir -p /server/install

COPY lighttpd.conf /server/install/.
COPY script_lighttpd.sh /server/install/.
RUN chmod +x /server/install/script_lighttpd.sh

ENTRYPOINT ["/server/install/script_lighttpd.sh", "/server/install/lighttpd.conf"]
