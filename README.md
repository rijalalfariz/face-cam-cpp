# **LiveFaceReco_RaspberryPi**

```
Attendance System Using Face Recognition with NCNN
```

This projects run in Raspberry pi 3 up to 10+ fps, connected with django-restframework to MIS website for attendance maagement and information built with django.

---

## Dependency

- OpenCV
- libcurl
- jsoncpp

---

## Run

change  `src/livefacereco.hpp(45) > project_path` to your own

```shell
mkdir build
cd build
cmake ..
make -j4
./LiveFaceReco
```

---

## Reference

- Neural Network Inference

  [ncnn](https://github.com/Tencent/ncnn)

- Detection:

  [mtcnn](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)

- Recognition: 

  [MobileFaceNet](https://github.com/deepinsight/insightface/issues/214)

-  Anti-Spoofing:

  [Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)

- Similar Face Recognition Project:
  
  [LiveFaceReco_RaspberryPi](https://github.com/XinghaoChen9/LiveFaceReco_RaspberryPi)