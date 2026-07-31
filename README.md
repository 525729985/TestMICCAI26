### **Run Infer**
Run script:
```bash
python run_inference.py
```
Or run docker:
```bash
sh build.sh
docker run \
  --shm-size=1G \
  --gpus all \
  --name thisxu \
  --rm \
  -v "$(pwd)/input_folder":/inputs \
  -v "$(pwd)/output_folder":/outputs \
  thisxu:latest
```