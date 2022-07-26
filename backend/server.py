import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from starlette.responses import StreamingResponse
import base64
app = FastAPI()


@app.post("/colorize")
async def function(file: UploadFile=File(...)):
    uploaded_file = "uploadedfile.png"
    filetoSave = await file.read()
    file = open(uploaded_file, "wb")
    file.write(filetoSave)
    file.close()
    prototxt_path = "experimentation/src/models/colorization_deploy_v2.prototxt"
    model_path = "experimentation/src//models/colorization_release_v2.caffemodel"
    kernel_path = "experimentation/src//models/pts_in_hull.npy"
    image_path = uploaded_file


    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    points = np.load(kernel_path)


    points = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1,313], 2.606, dtype="float32")]

    #LAB -> L = Lightness a* b* color values
    bw_image = cv2.imread(image_path)
    normalized = bw_image.astype("float32") / 255.0
    lab  = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    (H_orig,W_orig) = bw_image.shape[:2] 
    resized = cv2.resize(lab, (224, 244))
    L = cv2.split(resized)[0]
    L -= 50
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0,:,:,:].transpose((1,2,0)) 

    (H_out,W_out) = ab.shape[:2]
    ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
    L = cv2.split(lab)[0]

    colorized = np.concatenate((L[:,:, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (255.0 * colorized).astype("uint8")
    
    cv2.imwrite("output.png", colorized)
    with open("output.png", "rb") as f:
        contents = f.read()
    return StreamingResponse(io.BytesIO(contents), media_type="image/png")
    
