FROM python:3.10-slim
WORKDIR /usr/src/app
COPY . .
# Install system dependencies (if needed, e.g., libgl for image processing)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install  numpy opencv-python gradio imageio
EXPOSE 7866
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7866

CMD ["python", "app.py"]