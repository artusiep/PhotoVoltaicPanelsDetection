Download data for experiments data using

DOWNLOAD
```
gsutil -m rsync -r gs://photo-voltaic-panels-detection/data/ data
```
## Result
UPLOAD
```
gsutil -m rsync -r data/result gs://photo-voltaic-panels-detection/data/result
```
## Thermal Panels
UPLOAD
```
gsutil -m rsync -r data/thermal-panels gs://photo-voltaic-panels-detection/data/thermal-panels
```
## Thermal Modules
UPLOAD
```
gsutil -m rsync -r data/thermal-modules gs://photo-voltaic-panels-detection/data/thermal-modules
```