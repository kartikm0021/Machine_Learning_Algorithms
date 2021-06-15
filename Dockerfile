FROM debian:buster

ENV LANG C.UTF-8

RUN apt-get update -qq && \
    apt-get dist-upgrade -qq -y --no-install-recommends && \
    apt-get install -qq -y libpcre3 libgmp10 libssl-dev --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /usr/local/bin/
COPY duckling-example-exe .

EXPOSE 8000

CMD ["duckling-example-exe", "-p", "8000", "--no-access-log", "--no-error-log"]