FROM registry.app.unistra.fr/sertit/eo-containers:geo_sertit_latest


MAINTAINER Bastien Coriat
RUN pip install pysheds
ENV AWS_ACCESS_KEY_ID=***REMOVED***
ENV AWS_SECRET_ACCESS_KEY=***REMOVED***
ENV USE_S3_STORAGE=1
ENV PYTHONPATH=/usr/lib/python3/dist-packages/

RUN apt-get update && apt-get install -y python3-gdal wget
RUN wget https://f-tep.com/sites/default/files/pythonapi/ftep_api-0.0.16-py3-none-any.whl \
    && wget https://f-tep.com/sites/default/files/ftep_util-0.0.8-py3-none-any.whl \
    && pip install ftep_api-0.0.16-py3-none-any.whl ftep_util-0.0.8-py3-none-any.whl \
    && rm -f ftep_api-0.0.16-py3-none-any.whl ftep_util-0.0.8-py3-none-any.whl

RUN pip install numpy==1.25.2

# Prepare processor script
RUN mkdir -p /home/worker/processor
COPY * /home/worker/processor/
# ENTRYPOINT ["/home/worker/processor/workflow.sh"]
