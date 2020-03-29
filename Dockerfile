
FROM ubuntu:bionic

ENV SUMO_HOME=/usr/share/sumo

RUN mkdir /app
WORKDIR /app

COPY --chown=root ./system-repositories-linux.txt /app
RUN apt-get update -y && \
    apt-get install -y software-properties-common;
RUN xargs -L1 add-apt-repository <./system-repositories-linux.txt

COPY --chown=root ./system-requirements-linux.txt /app
RUN apt-get update -y && \
    xargs apt-get install -y <./system-requirements-linux.txt

COPY --chown=root ./requirements.txt /app
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

ENV REGIONS_PATH=/.regions
ENV SCENARIO_PATH=/app/scenario
ENV OUTPUT_PATH=/app/output

COPY --chown=root . /app

RUN groupadd -r traffic-light-group && useradd --no-log-init -r -g traffic-light-group traffic-light

RUN chmod -R a=rwx /app

USER traffic-light

VOLUME ${REGIONS_PATH}
VOLUME ${SCENARIO_PATH}
VOLUME ${OUTPUT_PATH}

ENTRYPOINT ["python3"]
CMD ["./traffic-light-optimization/sumo_main.py"]
