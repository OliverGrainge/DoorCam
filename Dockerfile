FROM python:3.9

COPY . /app



RUN python -m venv venv
ENV PATH="/usr/src/app/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
CMD ["python",  "./train.py"]