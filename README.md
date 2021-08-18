# **Face Recognition Attendance System**

```
Attendance System Using Face Recognition with NCNN
```

This projects run in Raspberry pi 3 up to 10+ fps, connected with django-restframework to MIS website for attendance maagement and information built with django (see [MIS Website Repo](https://github.com/rijalalfariz/face-attendance-admin)).

---

## Dependencies

- OpenCV
- libcurl
- jsoncpp

---

## Run

Camera Runs in Raspbian OS on Raspberry Pi 3B (or better)

Install Dependencies
Run Webserver (see [MIS Website Repo](https://github.com/rijalalfariz/face-attendance-admin))
Add some employee picture on webserver (it sends to `img/` folder) to try recognition performance
change `src/livefacereco.cpp(83, 312, 450) > YOUR_HOST` to your own host (for localhost, use your computer's ip address)
change  `src/livefacereco.hpp(45) > project_path` to your own

After step above, run:
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