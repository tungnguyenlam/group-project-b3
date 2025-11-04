FROM continuumio/miniconda3:25.3.1-1

WORKDIR /bubble-segmentation-final-deep-learning

RUN apt-get update && apt-get install -y git curl wget unzip && apt-get clean
RUN pip install gdown

# Copy only the environment file first for better caching
COPY ./environments/ /environments/

EXPOSE 8080

CMD ["bash", "-c", "chmod +x /environments/set_up.sh && /environments/set_up.sh && conda run -n py11 jupyter lab --ip=0.0.0.0 --port=8080 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"]
