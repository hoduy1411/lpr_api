FROM python:3.8.14-slim-buster

WORKDIR /code

# Install opencv
ENV TZ="Asia/Ho_Chi_Minh"
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* 

    # set env variables
    ENV PYTHONDONTWRITEBYTECODE 1
    ENV PYTHONUNBUFFERED 1
    
    COPY ./requirements.txt /code/requirements.txt
    
    # Install Python dependencies
    RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install cryptography
RUN apt-get update && apt-get install -y build-essential

COPY ./app /code/app
COPY ./cfg /code/cfg
COPY ./models /code/models
COPY ./run.py /code/run.py
COPY ./.env /code/.env
COPY ./build.sh /code/build.sh
# compile and clean code
RUN chmod +x build.sh && ./build.sh

# Run the application
ENTRYPOINT ["./run.bin"]