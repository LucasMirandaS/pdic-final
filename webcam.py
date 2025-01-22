import cv2
from ultralytics import YOLO

model = YOLO('./runs/detect/train2/weights/best.pt')
print(model.names)
webcamera = cv2.VideoCapture(1)

valores_moedas = {
    0: 0.1,
    1: 1,
    2: 0.25,
    3: 0.05,
    4: 0.50,
}

while True:
    success, frame = webcamera.read()
    
    results = model(frame, conf=0.4, imgsz=480)
    contagem_moedas = 0

    if (len(results[0]) > 0 ):
        result = results[0]
        boxes = result.boxes
        classes = boxes.cls

        if boxes is not None:
            for i in range(len(boxes)):
                cls = int(classes[i])
                print(cls)
                valor = valores_moedas.get(cls, 0)
                contagem_moedas+=valor

    cv2.putText(frame, f"Moedas: {len(results[0])} | Valor total: R$ {contagem_moedas:0.2f}" , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Camera", results[0].plot())

    if cv2.waitKey(1) == ord('q'):
        break

webcamera.release()
cv2.destroyAllWindows()
