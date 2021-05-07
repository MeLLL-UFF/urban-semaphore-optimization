
FROM tensorflow/tensorflow:2.1.0-gpu

ARG USER_ID
ARG GROUP_ID

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

RUN curl https://sumo.dlr.de/releases/1.8.0/sumo-src-1.8.0.tar.gz \
    | tar -xz
ENV SUMO_HOME=sumo-1.8.0
RUN mkdir sumo-1.8.0/build/cmake-build && cd sumo-1.8.0/build/cmake-build \
    && cmake ../.. && make -j$(nproc)

ENV REGIONS_PATH=/.regions
ENV SCENARIO_PATH=/app/scenario

ENV FRAP_DATA_PATH=/app/traffic-light-optimization/algorithm/frap/internal/frap_pub/data
ENV FRAP_MODEL_PATH=/app/traffic-light-optimization/algorithm/frap/internal/frap_pub/model
ENV FRAP_RECORDS_PATH=/app/traffic-light-optimization/algorithm/frap/internal/frap_pub/records
ENV FRAP_SUMMARY_PATH=/app/traffic-light-optimization/algorithm/frap/internal/frap_pub/summary

COPY --chown=root . /app

RUN addgroup --gid $GROUP_ID user && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
RUN chown -R $USER_ID:$GROUP_ID /app && chmod -R a=rwx /app

USER user

VOLUME ${REGIONS_PATH}
VOLUME ${SCENARIO_PATH}

VOLUME ${FRAP_DATA_PATH}
VOLUME ${FRAP_MODEL_PATH}
VOLUME ${FRAP_RECORDS_PATH}
VOLUME ${FRAP_SUMMARY_PATH}


ENTRYPOINT ["python3"]
CMD ["./traffic-light-optimization/sumo_main.py"]
