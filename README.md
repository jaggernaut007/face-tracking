# face-tracking
Track known and unknown face.

## Features
1) This script turns your webcam on, using IPWebcam.It then scans the video feed and recognizes the faces who are already registered before. 
2) It then waits for faces to show up on the video feed.
3) If a face is already in the database, It shows the name of the person 
4) Otherwise if the user who has just shown up on the feed is new to the system, it trains their face data instantly to the system and recognizes them immediately afterwards and henceforth at the same time adding their details to the CSV DB. 

## Known bugs ## TODO

1) Lag when training 
2) Slower script because of Disabled tracking.The script recognizes on every frame at the moment
3) Only one user at a time
4) Add support for all webcams
5) Add support for a better level 2 training  


## Installation 
1) Download All the  files from the zip file
2) (Optional) Add custom training data to images folder under the users name for better accuracy
4) Install all imports
3) Run faces_train.py
4) Run faces.py for tracking faces.

Let me know if I might have skipped some issues with the script/installation/bugs

