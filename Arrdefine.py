from spyne import Unicode, Array, ComplexModel,Float,String

class Req_Image(ComplexModel):
    imgcontent = Unicode
    imgID = Unicode
    equiptype = String

class Img_Rectangle (ComplexModel):
    xmin = int
    ymin = int
    xmax = int
    ymax = int
    width = int
    height = int

class Reco_Equipment_child (ComplexModel):
    equiName = String
    area = Img_Rectangle
    state = Unicode
    score = Unicode

class Reco_Equipment (ComplexModel):
    equiName = String
    area = Img_Rectangle
    acreage = int
    state = Unicode
    score = Unicode
    children = Array(Reco_Equipment_child)

class Res_Image(ComplexModel):
    imgID = Unicode
    equipments = Array(Reco_Equipment)

class Req_Model(ComplexModel):
    modelConf = Unicode
    model = Unicode
    isstart = Unicode
    isend = Unicode

class Res_model(ComplexModel):
    updModRes = Unicode

