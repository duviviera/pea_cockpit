# Dockerfile
# Stage 1: Build Stage
FROM python:3.13-slim as builder

# Define the target installation directory inside the container
ENV INSTALL_PATH=/usr/local/lib/python3.13/site-packages

# Install uv
RUN pip install uv

# Set the working directory
WORKDIR /app

# Copy configuration files
COPY pyproject.toml .
COPY uv.lock .

# Run uv sync, explicitly telling it to install dependencies into the defined path.
RUN uv pip install --target ${INSTALL_PATH} .

# Stage 2: Final Runtime Stage
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Set the PYTHONPATH to the exact location where we installed packages in Stage 1.
ENV PYTHONPATH=/usr/local/lib/python3.13/site-packages

# Copy the installed packages from the builder stage into the final stage's PYTHONPATH location.
# This must match the INSTALL_PATH/PYTHONPATH.
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages

# Copy the application source code and data folders
COPY src src
COPY data data
COPY app.py .
COPY README.md .

# Set the entry point using the 'python -m' standard method
EXPOSE 8501
ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]