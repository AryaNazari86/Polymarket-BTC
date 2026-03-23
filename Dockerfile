FROM python:3.9-slim

# OpenMP for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY collect_features.py train.py signal_bot.py combined_signal_bot.py polymarket_collector.py ./

# Directories for output
RUN mkdir -p /app/models /app/odds_data

CMD ["bash"]
