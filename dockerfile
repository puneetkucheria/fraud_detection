# 
FROM python:3.8

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./app.py /code/app.py
COPY ./model_building.py /code/model_building.py
COPY ./model_classifier.pkl /code/model_classifier.pkl

# 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]