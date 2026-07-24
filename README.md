### **Infer**
    ```bash
    docker run \
      --shm-size=1G \
      --gpus all \
      --name thisxu \
      --rm \
      -v "$(pwd)/input_folder":/inputs \
      -v "$(pwd)/output_folder":/outputs \
      thisxu:latest
    ```