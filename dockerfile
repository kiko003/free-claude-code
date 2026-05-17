FROM python:3.11

WORKDIR /app

COPY . .

# Install uv
RUN pip install uv

# Install dependencies (force correct env)
RUN uv sync

# Activate uv environment + run app
CMD ["uv", "run", "server.py"]

EXPOSE 8082