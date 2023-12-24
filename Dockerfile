FROM python:3.10

RUN mkdir -p /workspace

WORKDIR /workspace

RUN pip3 install torch torchvision 

RUN pip install torch_geometric networkx[default] matplotlib
