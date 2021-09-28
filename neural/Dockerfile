FROM tensorflow/tensorflow:latest-gpu
WORKDIR /root

# Installs pandas, google-cloud-storage, and scikit-learn
# scikit-learn is used when loading the data
RUN /usr/bin/python3 -m pip install --upgrade pip
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN pip install gsutil

# Copies the trainer code to the docker image.
COPY learning_model.py ./learning_model.py
COPY models_builders.py ./models_builders.py
COPY models ./models
COPY utils ./utils
COPY convert.py ./convert.py
COPY run_learning.sh ./run_learning.sh

ENTRYPOINT ["bash", "run_learning.sh"]