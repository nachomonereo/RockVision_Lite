from roboflow import Roboflow

rf = Roboflow(api_key="vr0Fi7dqHrIR5Zypf2sy")
project = rf.workspace("bandrma-onyedi-eyll-niversitesi-byubb").project("rock-classification-bvis1")
version = project.version(2)
dataset = version.download("yolov5", location="datasets/rock_classification")


