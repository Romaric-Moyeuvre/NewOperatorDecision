# - - - - - - - - - - - - - - - - - - -
FROM ubuntu:22.04 as python-base

ARG PYTHON_VERSION
ARG FORCE_USER_ID
ARG FORCE_GROUP_ID
ARG PACKAGE_NAME
ARG POETRY_VERSION

ENV PACKAGE_NAME=${PACKAGE_NAME}
ENV PYTHON_VERSION=${PYTHON_VERSION}
ENV POETRY_VERSION=${POETRY_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV POETRY_VIRTUALENVS_CREATE=false

USER root

RUN apt-get update \
    && apt-get --yes install apt-utils \
    && apt-get --yes install curl wget

RUN apt-get --yes install python3 python3-pip python3-venv
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN curl https://repo.anaconda.com/pkgs/misc/gpgkeys/anaconda.asc | gpg --dearmor > conda.gpg
RUN install -o root -g root -m 644 conda.gpg /usr/share/keyrings/conda-archive-keyring.gpg
RUN gpg --keyring /usr/share/keyrings/conda-archive-keyring.gpg --no-default-keyring --fingerprint 34161F5BF5EB1D4BFBBB8F0A8AEB4F8B29D82806
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/conda-archive-keyring.gpg] https://repo.anaconda.com/pkgs/misc/debrepo/conda stable main" > /etc/apt/sources.list.d/conda.list

RUN apt-get update
RUN apt-get --yes upgrade

RUN apt-get --yes install conda

ENV PATH=${PATH}:/opt/conda/bin

RUN conda install python=${PYTHON_VERSION}

RUN conda init bash
# RUN conda install -c conda-forge poetry==${POETRY_VERSION}

## add a user with same GID and UID as the host user that owns the workspace files on the host (bind mount)
RUN groupadd -f servicegroup -g ${FORCE_GROUP_ID}
RUN useradd -s $(which bash) --uid ${FORCE_USER_ID} --gid ${FORCE_GROUP_ID} -m serviceuser

USER serviceuser

RUN mkdir ~/app
ENV PATH=${PATH}:/opt/conda/bin
RUN conda init bash
RUN conda create -n env-${PACKAGE_NAME} python=${PYTHON_VERSION}
RUN echo "export PATH=$PATH:/opt/conda/bin:~/.local/bin"  >> ~/.bashrc
RUN echo "conda activate env-${PACKAGE_NAME}" >> ~/.bashrc
RUN echo "cd ~/app"  >> ~/.bashrc

# - - - - - - - - - - - - - - - - - - -
FROM python-base as runtime

ARG INTERNAL_PORT

USER serviceuser
WORKDIR /home/serviceuser/app

# Install requirements
COPY app/requirements.txt .
COPY app/shiny-frontend .

# RUN ["bash", "-c", "conda install pip"]
# RUN ["bash", "-c", "pip install --no-cache-dir --upgrade -r requirements.txt"]

RUN pip install --no-cache-dir --upgrade -r requirements.txt
# RUN conda install --file requirements.txt
# RUN [ ! -f pyproject.toml ] || poetry install --no-interaction --no-ansi -vvv

# ENTRYPOINT ["bash"]
# CMD ["uvicorn", "serviceuser:servicegroup", "--host", "0.0.0.0", "--port", ${INTERNAL_PORT}]

CMD ["/home/serviceuser/.local/bin/shiny", "run", "--reload", "--host", "0.0.0.0", "--port", "3838", "shiny-frontend/app.py"]
