from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("weights/best.pt")  # load a custom trained model

    # Export the model
    model.export(format="onnx")