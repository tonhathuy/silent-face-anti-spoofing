# face anti spoofing
## requirement
```bash
docker pull 0105200696/silent-face-anti-spoofing:v2
```
## RUN 
```bash
git clone https://github.com/tonhathuy/silent-face-anti-spoofing.git
cd silent-face-anti-spoofing
python app_v2.py
``` 
## config 
* SERVICE_IP: "0.0.0.0"
* PORT: 5002
* MODEL: "./model/anti_spoof_models"
* DEVICE_ID: 0
