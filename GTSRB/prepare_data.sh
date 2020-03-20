mkdir datasets
cd datasets
mkdir gtsrb
cd gtsrb
wget https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip
unzip traffic-signs-data.zip
cd ../..
python generate_gtsrb_data.py
