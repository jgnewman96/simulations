FROM python:3.7

# Install python dependencies
COPY requirements.txt /
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN rm requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/src/"
WORKDIR /src/