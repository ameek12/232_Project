curl -O https://downloads.rclone.org/v1.58.0/rclone-v1.58.0-linux-amd64.zip
unzip rclone-v1.58.0-linux-amd64.zip
rm rclone-v1.58.0-linux-amd64.zip

rclone-v1.58.0-linux-amd64/rclone config
rclone-v1.58.0-linux-amd64/rclone sync openaq:openaq-dat-archive ~/cmerry/data

~/cmerry/data/records/csv.gz -type f -name "*.gz" -exec gunzip {} +
