from ultralytics import YOLO

#model = YOLO("yolov8n.yaml")
#model.train(data="numbers.yaml", epochs=50)

# Load a model
model = YOLO('runs/detect/train2/weights/last.pt')  # load a partially trained model

# Resume training
results = model.train(resume=True)