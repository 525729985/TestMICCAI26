### **Infer**
    ```bash
    docker run \
      --shm-size=1G \
      --gpus all \
      --name thisxu \
      --rm \
      -v "./input_folder":/inputs \
      -v "/output_folder":/outputs \
      thisxu:latest
    ```