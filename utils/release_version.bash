docker stop $(docker ps -aq) && docker rm $(docker ps -aq)
docker build --tag pablogod/fishv4:1.11 .
docker run --publish 8501:8501 --detach pablogod/fishv4:1.11
docker run --publish 8501:8501 pablogod/fishv4:1.11
docker push pablogod/fishv4:1.11

streamlit run app.py