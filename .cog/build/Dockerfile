#syntax=docker/dockerfile:1.4
FROM r8.im/cog-base:cuda12.6-python3.12-torch2.6.0
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /usr/local/bin/
ENV UV_SYSTEM_PYTHON=true
COPY --from=cog_build requirements.txt /tmp/requirements.txt
ENV CFLAGS="-O3 -funroll-loops -fno-strict-aliasing -flto -S"
RUN --mount=type=cache,target=/root/.cache/pip uv run pip install --cache-dir /root/.cache/pip -r /tmp/requirements.txt
ENV CFLAGS=
ENV CFLAGS="-O3 -funroll-loops -fno-strict-aliasing -flto -S"
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --no-cache cog
ENV CFLAGS=
RUN pip install flash-attn==2.7.4.post1 --no-build-isolation
WORKDIR /src
EXPOSE 5000
ENV COG_PREDICT_TYPE_STUB="predict.py:Predictor"
CMD ["python", "-m", "cog.server.http"]
COPY . /src