# Build from the repo root:  docker build --platform linux/amd64 -f mesh/Dockerfile -t <image> .
# Base images come from mirror.gcr.io (Google's mirror of the official Docker Hub images): Docker Hub rate-limits anonymous pulls.
FROM mirror.gcr.io/library/python:3.13-slim
# PyPy for the sandboxed simulator worker only (generated simulators + search run ~4-5x faster); the protocol
# process stays on CPython, and a simulator that does not load under PyPy falls back to CPython (mesh/simgen.py).
COPY --from=mirror.gcr.io/library/pypy:3.10-slim /opt/pypy /opt/pypy
# A plain English word list (/usr/share/dict/words, ~1 MB, Debian `wamerican`) for word games: candidate answers are
# checked inside the container against the simulator, so no game content ever leaves it (rule 7.11).
RUN apt-get update && apt-get install -y --no-install-recommends wamerican && rm -rf /var/lib/apt/lists/*
# Word frequencies (wordfreq, data shipped in the package, no network at run time): in word games the common words are
# tried first. The only pip package in the image; the harness itself stays standard library.
RUN pip install --no-cache-dir "wordfreq==3.1.1"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /bot
COPY mesh/ /bot/mesh/
# -u: unbuffered; the shell flushes every protocol line anyway. Stdlib only: no pip installs.
CMD ["python", "-u", "-m", "mesh"]
