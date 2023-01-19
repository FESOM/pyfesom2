FROM mambaorg/micromamba
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
WORKDIR /app
# Yeah, this next one is dumb. But it seems to be a requirement either in
# Docker or in Mamba, Paul can't tell which but this "does the trick":
RUN sed -i 's/name: pyfesom2/name: base/g' ./ci/requirements-py37.yml
RUN micromamba install -f ./ci/requirements-py37.yml && \
  micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip install .
RUN which -a pfplot pfinterp

