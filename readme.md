# Video matcher

This script will help you to rescue a video edition project you have  have accidently lost in the crash of your computer.
It assume that you have both 
- a mid-quality version of the edited video you may have send to your collaborators
- the rushes you have used to create that video

It will then recreate automatically the timeline for your edited video by comparing each frame of the edited video with a dense subsets of the frames (every 10 frames) in the rush and create a movie with both a low resolution version of the edited video and the found corresponding frames in the rush, with the path to the rush video, its name and the matching frame number in the rush video as shown below. The name will appear in blue if this is considered by the algorithm to be reliable match, otherwise the name will appear in black.

![alt tag](https://raw.github.com/martinGithub/videomatcher/master/example.png)

# Install

## windows



* install the python with packages numpy, matplotlib, scipy, python-opencv ans scikit-learn
 * you can install [winpython](http://winpython.github.io/) or [pytonh(x,y)](http://python-xy.github.io/) that will install python with numpy, matplotlib and scipy.  
 * you can install separetly python and the various packages  
* download the zip file of the project and decompress it 
* double click on *videomatcher.py* (You will have to *register* winpython as explained on its installation guide it to be able to launch the script by double clicking on it)

## mac os

* install python from https://www.python.org/ftp/python/2.7.10/python-2.7.10-macosx10.6.pkg (note that recent version of Mac OS X should comes with Python 2.7 out of the box)
* install the python packages numpy, matplotlib, scipy, python-opencv ans scikit-learn
* download the zip file of the project and decompress it 
* in the terminal, move in the decompressed folder using the *cd* command (for example *cd ~/Downloads/videomatcher* if that where it has been decompressed) and type *python videomatcher.py*,

## ubuntu

* install the python packages numpy, matplotlib, scipy, python-opencv ans scikit-learn
* download the zip file of the project and decompress it 
* in the terminal, move in the decompressed folder and type *python videomatcher.py*

# Using the script

Run the script from the command line using
python videomatcher.py
The script will ask you in this order for 
- the folder containing the rushes
- the video file containing side  edited video
- the name of the video that  will be generated with side by side display of the matching frames
- the folder in which small images extracted from the rushed will be stored (you need to make sure that the hard drive has  enough space to store these images)

The scipt will generate:
- the video  with side by side display of the matching frames
- a file timeline.txt that tells you which rush is used in which part of the video
- matched_frame.pkl , which is binary file used by the script to store matchings frames and matching scores.

# How this works

The script will first run trough all the rush videos in the folder you have provided and will extract one frame every ten frames (3 per seconds if you have videos recoreded at 30fps) and save a low resultion image for each of these frames. The name if each of these images is chosent to correspond to the concatenation of path the rush video, the video name and the frame number. This will will create a database of small images, that can be quite large if you have a lot of rushes.

Then the scipt will take sequentialy each frame of the edited video and will look for the most similar image in the database of small images. This is done efficently by using an approximate nearest neighboor algorithm that accelerate the search for similar images. 
Due to the fact that we kept only one frame every 10 frames in the rushes, you can expect only 1 frame every 10 frame of the edited wideo to get a perfect match, which is in general sufficient to reconstruct your timeline.

# Limitations

* it wills not work for frames that have been either rotated, mirrored, or if you have modified the color balance.
This could be improved.
* the entire set of small images is stored in the RAM during the matching phase, this may be too big if you have a lot of rushes and crash you computer.

# Possible improvements

* We could use a adaptive frame selection in the rush to avoid the constant 10 frames steps. The idea would be to sample more frame when there are large motions in the video.
* we could avoid putting all the frame in the RAM duing nearest neighborr search.
* we coudl use temporal coherence and try to fill gaps between found matches by reopening the found rush videos and extracting new frames.



  
