# Download base image ubuntu 18.04
FROM ubuntu:18.04

# streamlit-specific commands for config
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p ~/.streamlit

RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > ~/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > ~/.streamlit/config.toml'

# install Python and Pip
# NOTE: libSM.so.6 is required for OpenCV Docker
# or you will get seg fault when import OpenCV
# error libGL.so.1
RUN apt-get update && \
    apt-get install -y \
    python3.7 python3-pip \
    ffmpeg libsm6 libxext6 libxrender-dev libgl1-mesa-dev

# expose port 8501 for streamlit
EXPOSE 8501

# make app directiry
WORKDIR /streamlit-docker

# upgrade for new versions of opencv
RUN pip3 install --upgrade pip

# copy requirements.txt
COPY requirements.txt ./requirements.txt

# install dependencies
RUN pip3 install -r requirements.txt

# copy all files over
COPY . .

# set heroku_startup.sh to be executable
#RUN chmod +x ./heroku_startup.sh

# download YOLO weights
RUN gdown --output ./fishv4/fish.weights --id 1_Uo_jB4ZVsBRA7I3MLisq_GZ8EQHeWr9

# launch streamlit app
CMD streamlit run --server.enableCORS false --server.port $PORT app.py