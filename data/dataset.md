# Dataset Description  

## Sample Data  
| Plate | Longitude  | Latitude  | Time                | Status |
| ----- | ---------- | --------- | ------------------- | ------ |
| 4     | 114.10437  | 22.573433 | 2016-07-02 00:08:45 | 1      |
| 1     | 114.179665 | 22.558701 | 2016-07-02 00:08:52 | 1      |
| 0     | 114.120682 | 22.543751 | 2016-07-02 00:08:51 | 0      |
| 3     | 113.93055  | 22.545834 | 2016-07-02 00:08:55 | 0      |
| 4     | 114.102051 | 22.571966 | 2016-07-02 00:09:01 | 1      |
| 0     | 114.12072  | 22.543716 | 2016-07-02 00:09:01 | 0      |

This is how the data looks like. In the data/ folder, each .csv file is trajectories for 5 drivers on the same day. Each trajectory step is detailed with features such as longitude, latitude, time, and status. 
Data can be found here in this [Google Drive](https://drive.google.com/file/d/1xfyxupoE1C5z7w1Bn5oPRcLgtfon6xeT/view).  

## Feature Description
- Plate: Plate means the taxi's plate. In this project, we change them to 0~5 to keep anonymity. The same plate means the same driver, so this is the target label for the classification.  
- Longitude: The longitude of the taxi.  
- Latitude: The latitude of the taxi.  
- Time: Timestamp of the record.  
- Status: 1 means the taxi is occupied and 0 means a vacant taxi.

