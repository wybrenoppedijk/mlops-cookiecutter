#base image
FROM python:3.7-slim
RUN apt update && \
   apt install --no-install-recommends -y build-essential gcc && \
   apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY model/ model/

WORKDIR /
RUN python3 -m pip install -r /tmp/requirements.txt
ENTRYPOINT ["python", "-u", "src/models/predict_model.py", "models/mnistcorrupted/model.pt", "data/processed/corruptmnist/test.pt"]