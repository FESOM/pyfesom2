FROM mambaorg/micromamba
COPY --chown=$MAMBA_USER:$MAMBA_USER . /tmp
RUN micromamba install -f ./ci/requirements-py37.yml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip install .
RUN which -a pfplot pfinterp

