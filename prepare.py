from roboflow import Roboflow

rf = Roboflow(api_key="CHAVE DE API")
project = rf.workspace("brazilian-coin").project("brazilian-coins-djfha")
version = project.version(3)
dataset = version.download("yolov8")
