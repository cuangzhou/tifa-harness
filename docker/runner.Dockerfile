FROM python:3.12-alpine@sha256:aa679aa4eed6eb56c1dc6ad3f1b98b7d2d788fd961596779d188fdedad97fb38
RUN addgroup -g 65532 tifa && adduser -D -u 65532 -G tifa tifa
USER 65532:65532
WORKDIR /workspace
ENTRYPOINT []
