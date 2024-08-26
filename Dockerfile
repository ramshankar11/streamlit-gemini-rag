FROM python:3.9-slim
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . ./
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--ui.hideTopBar=True","--client.toolbarMode=minimal","--global.developmentMode=False","--server.address=0.0.0.0"]
