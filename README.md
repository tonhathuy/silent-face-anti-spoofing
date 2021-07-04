# face anti spoofing
## requirement
```bash
docker pull 0105200696/silent-face-anti-spoofing:v2
```
## RUN 
```bash
git clone https://github.com/tonhathuy/light-weight-face-anti-spoofing.git
cd light-weight-face-anti-spoofing
python service.py
``` 
## config 
* SERVICE_IP: "0.0.0.0"
* PORT: 5003
* MODEL: " ./model/spf_models/MN3_antispoof.xml"
* DEVICE_ID: 0
