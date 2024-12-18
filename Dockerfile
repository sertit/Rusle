FROM sertit/eo-containers:geo2

MAINTAINER Bastien Coriat
ENV AWS_ACCESS_KEY_ID=***REMOVED***
ENV AWS_SECRET_ACCESS_KEY=***REMOVED***
ENV AWS_S3_ENDPOINT=s3.waw3-1.cloudferro.com
ENV USE_S3_STORAGE=1

RUN apt-get update && apt-get install -y wget libgdal-dev build-essential
RUN pip install gdal==3.6.2
RUN wget https://f-tep.com/sites/default/files/pythonapi/ftep_api-0.0.16-py3-none-any.whl \
    && wget https://f-tep.com/sites/default/files/ftep_util-0.0.8-py3-none-any.whl \
    && PYTHONPATH=/usr/lib/python3/dist-packages/ pip install ftep_api-0.0.16-py3-none-any.whl ftep_util-0.0.8-py3-none-any.whl \
    && rm -f ftep_api-0.0.16-py3-none-any.whl ftep_util-0.0.8-py3-none-any.whl

RUN pip install --index-url https://__token__:glpat-xypsdgKBaMHiRvqraGj1@git.unistra.fr/api/v4/projects/26129/packages/pypi/simple rusle==2.3.2
RUN pip install sertit==1.42.3
RUN pip install numpy==1.26.4
# Prepare processor script
RUN mkdir -p /home/worker/processor
# COPY * /home/worker/processor/
# ENTRYPOINT ["/home/worker/processor/workflow.sh"]