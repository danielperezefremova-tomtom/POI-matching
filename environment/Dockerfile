# ToDo: Version would be the tag
FROM mapsanalytics.azurecr.io/databricks:9.1
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

EXPOSE 8000
RUN sudo apt update && sudo apt-get install -y git && sudo apt-get -y install python3.8-venv
RUN git config --global --add safe.directory /workspace

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PIP_DEFAULT_TIMEOUT=100

COPY /src/requirements.in /tmp/requirements.in
RUN pip install pip-tools  \
    && pip-compile --resolver legacy /tmp/requirements.in > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt \
    && pip install -U ipykernel

USER $USERNAME
