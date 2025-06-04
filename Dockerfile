FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY dashboard.py api.py tca2.png ./
COPY startup.sh ./

RUN chmod +x startup.sh

EXPOSE 8000 8501

CMD ["./startup.sh"]