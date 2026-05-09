
from roboflow import Roboflow
rf = Roboflow(api_key="XbykWsKuFqKZL7IrUCL8")
project = rf.workspace("luis-lheslie-judilla-lmona").project("emergency-vehicles-xug80")
version = project.version(12)
dataset = version.download("yolov8")
                