# Kuhn-NFSP training / evaluation environment.
# OpenSpiel has no reliable native Windows path, so every run happens in here.
#
# Build:  docker build -t kuhn-nfsp .
# Smoke:  docker run --rm kuhn-nfsp python -c "import pyspiel; g=pyspiel.load_game('kuhn_poker'); print(g.num_distinct_actions(), g.num_players())"

FROM python:3.11-slim

# Runtime libs the OpenSpiel / numpy / torch C++ extensions link against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv handles dependency installation (see requirements.txt).
RUN pip install --no-cache-dir uv

WORKDIR /app

# Dependencies first so the layer caches independently of source changes.
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Project source. Dev runs bind-mount the working tree over /app; this COPY
# keeps the image self-contained for a clean-clone reproduction.
COPY src/ ./src/
COPY configs/ ./configs/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

CMD ["python", "-c", "import pyspiel; print(pyspiel.load_game('kuhn_poker'))"]
