FROM ubuntu:latest

COPY /models /app/models
COPY /prediction /app/API
COPY startup.sh startup.sh
COPY requirements.txt requirements.txt

RUN apt update 
RUN apt-get update
RUN apt-get install
RUN apt install -y protobuf-compiler 
RUN apt-get install -y python3
RUN bash startup.sh
CMD [ "python3","app/API/main.py" ]

EXPOSE 8080/tcp
