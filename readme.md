# Video matcher

This script will help you to rescue a video edition project you have  have accidently lost in the crash of your computer and which you did not duplicated because you are not paranoid enough with computers. 
It assume that you have both 
- a mid-quality version of the edited video you may have send to your collaborators
- the rushes you used to create that video

It will then recreate automatically the timeline by comparing each frame of the edited video with all the frames in the rush and create a movie with both a low resolution version of the edited video and the found corresponding frames in the rush, with the path to the rush video,its name and the matching frame number in the rush video as shown below.

![alt tag](https://raw.github.com/martinGithub/videomatcher/master/example.png)

# Install

## windows

* install python from https://www.python.org/ftp/python/2.7.10/python-2.7.10.amd64.msi.
* install the python packages numpy, matplotlib, scipy, python-opencv ans scikit-learn
* download the zip file of the project and decompress it 
* double click on *videomatcher.py*

## mac os

* install python from https://www.python.org/ftp/python/2.7.10/python-2.7.10-macosx10.6.pkg (note that recent version of Mac OS X should comes with Python 2.7 out of the box)
* install the python packages numpy, matplotlib, scipy, python-opencv ans scikit-learn
* download the zip file of the project and decompress it 
* in the terminal, move in the decompressed folder using the *cd* command (for example *cd ~/Downloads/videomatcher* if that where it has been decompressed) and type *python videomatcher.py*,

## ubuntu

* install the python packages numpy, matplotlib, scipy, python-opencv ans scikit-learn
* * download the zip file of the project and decompress it 
* in the terminal, move in the decompressed folder and type *python videomatcher.py*

# Using the script

run the script from the command line using
python videomatcher.py
the script will ask you in this order for 
- the folder containing the rushes
- the video file containing side  edited video
- the name of the video that  will be generated with side by side display of the matching frames
- the folder in which small images extracted from the rushed will be stored (you need to make sure that the hard drive has
 enough space to store these images)

the scipt will generate:
- the video  with side by side display of the matching frames
- a file timeline.txt that tells you which rush is used in which part of the video
- matched_frame.pkl , which is binary file used by the script to store matchings frames and matching scores.




  
