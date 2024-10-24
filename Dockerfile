FROM ubuntu:latest
LABEL authors="davidwardan"

ENTRYPOINT ["top", "-b"]