from ultralytics import YOLO

model = YOLO("./runs/detect/train2/weights/best.pt")

results = model(["Captura de tela 2025-01-22 074134.png", "25_1477286292_jpg.rf.3c5f7c868154bf7edd16f7efaab09952.jpg", "5_1477145436_jpg.rf.3f5674eb85160183c1eb6085cb17eaaf.jpg"], conf=0.2)

for result in results:
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
    obb = result.obb
    result.show()
    result.save(filename="result.jpg")
