# Video matcher

This script will help you to rescue a video edition project you have  have accidently lost in the crash of your computer and which you did not duplicated because you are not paranoid enough with computers. 
It assume that you have both 
- a mid-quality version of the edited video you may have send to your collaborators
- the rushes you used to create that video

It will then recreate automatically the timeline by comparing each frame of the edited video with all the frames in the rush and create a movie with both a low resultion version of the edited video and the found corresponding frames in the rush, with the path to the rush video,its name and the frame number as shown below.

# Install

## windows
you need to first to install python 2.7
http://python.org/ftp/python/2.7.5/python-2.7.5.msi

http://docs.opencv.org/master/d5/de5/tutorial_py_setup_in_windows.html#gsc.tab=0

## mac os

## ubuntu

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
-the video that  will be generated with side by side display of the matching frames
-a file timeline.txt that tells you which rush is used in which part of the video
- matched_frame.pkl , which is binary file used by the script to store matchings frames and matching scores.




  
