FROM python:3.6
# set display port to avoid crash
ENV DISPLAY=:99

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
CMD ["main.py"]
