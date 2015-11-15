#! /usr/bin/env python
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import scipy.ndimage
import pickle
import time
import sklearn.neighbors
import shutil



def getPathsInteractive():
    """function to get all the file paths interactively"""
    import Tkinter, tkFileDialog
    root = Tkinter.Tk()
    rushesFolder = tkFileDialog.askdirectory(parent=root, initialdir="/",
                                             title='Please select the root directory that contains all the rushes')
    if len(rushesFolder) > 0:
        print "You chose %s" % rushesFolder

    videoFile = tkFileDialog.askopenfilename(parent=root, title='Choose the edited video file')

    myFormats = [
        ('Mpeg4', '*.mp4'),
    ]
    resultFolder = tkFileDialog.askdirectory(parent=root, filetypes=myFormats, title="Where do you want to save the matches results")

    thumbnails_folder = tkFileDialog.askdirectory(
        parent=root, initialdir="/",
        title='Please select the directory in which you want to save the miniaturized videos images'
    )
    if len(thumbnails_folder) > 0:
        print "You chose %s as the folder where you want to save the small images" % thumbnails_folder

    assert(rushesFolder != '')
    assert(videoFile != '')
    assert(thumbnails_folder != '')
    assert(resultFolder != '')
    
    return rushesFolder,videoFile,thumbnails_folder,resultFolder

def findAllRushVideos(rushesFolder):
    videoFiles=[]
    for root, _dirnames, _filenames in os.walk(rushesFolder):
        videoFiles.extend(glob.glob(root + "/*.mov"))
        videoFiles.extend(glob.glob(root + "/*.MOV"))
        videoFiles.extend(glob.glob(root + "/*.avi"))

    #removing potential symbolic links
    videoFiles2 = []
    for v in videoFiles:
        if os.path.isfile(v):
            videoFiles2.append(v)
            print 'added %s' % v
        else:
            print 'removing symbolic link %s'%v
    videoFiles = videoFiles2

    print 'found '+str(len(videoFiles)) +' videos'
 
    return videoFiles


def createAllThumbnails(thumbnails_folder,rushesFolder,videoFiles):
    try:
        os.mkdir(thumbnails_folder)
    except OSError:
        print "cannot create the folder"
    
    for i, videoFile in enumerate(videoFiles):
        basename = str.replace(os.path.relpath(videoFile, rushesFolder), '/', '|')
        print 'creating thumbnails for video %s (%d over %d)'%(basename, i+1, len(videoFiles))
        createThumbnails(videoFile, thumbnails_folder, basename=basename, skip=1)    

def main(rushesFolder,videoFile,thumbnails_folder,resultFolder):
    
    videoResult=os.path.join(resultFolder,'matches.avi')
    
    videoFiles=findAllRushVideos(rushesFolder)
    
    createAllThumbnails(thumbnails_folder,rushesFolder,videoFiles)




    
    bestframes, distances, _distancesMatrix,distancetosecondbest = retrieveImages(videoFile, thumbnails_folder)
    
    #save matches and scores into a binary file
    with open(os.path.join(resultFolder,'matched_frame.pkl'), 'wb') as f:
        d = dict()
        d['bestframes'] = bestframes
        d['distances'] = distances#
        d['distancetosecondbest']=distancetosecondbest
        d['distancesMatrix'] = None#distancesMatrix
        pickle.dump(d, f)

    with open(os.path.join(resultFolder,'matched_frame.pkl'), 'rb') as f:

        d = pickle.load(f)
        plt.figure()
        plt.plot(d['distances'])
        listAllSmallImages, database = loadAllSmallImageInMemory(thumbnails_folder)
        reduced_size = (database.shape[2], database.shape[1])
        videoTable = loadVideoSmall(videoFile, reduced_size)
        plt.figure()
        plt.imshow(np.zeros((videoTable.shape[1], videoTable.shape[2]*2)), cmap=plt.cm.Greys, vmin=0, vmax=255)

        #
        bestframes = database[d['bestframes'], :, :]
        distances = np.mean(np.abs(videoTable-bestframes), axis=(1, 2))
        distancetosecondbest=d['distancetosecondbest']
        movienames = []
        idframes = []

        for im in listAllSmallImages:
            idframes.append(int(im[-12:-4]))
            movienames.append(im[len(thumbnails_folder)+1:-18])

        with open(os.path.join(resultFolder,'all_frames.txt'), 'w') as f:
            #i=1
            #f.write('frame %d : %s at frame %d'%(i, movienames[i], idframes[i]))
            for i in range(d['bestframes'].size):
                b = d['bestframes'][i]
                f.write('frame %d : %s at frame %d\n'%(i, movienames[b], idframes[b]))

        # keep local minima
        combined_video = np.dstack((videoTable, bestframes))



        #keep = np.nonzero((distances[2:-2] < distances[0:-4]) \
                        #& (distances[2:-2] < distances[1:-3])
                        #& (distances[2:-2] < distances[3:-1])
                        #& (distances[2:-2] < distances[4:])
                        #& (distances[2:-2] < 15))[0]+2
                        
        keep=np.nonzero(distancetosecondbest>1.2*distances)[0]# keep only frame which have a smaller error that second best by some ratio

        keepbool = np.zeros(distances.shape, dtype=np.bool)
        keepbool[keep] = True
        zoom_factor = 4
        new_size = (combined_video.shape[2]*zoom_factor, combined_video.shape[1]*zoom_factor+20)


        video_writer = cv2.VideoWriter(videoResult, cv2.cv.FOURCC(*"MJPG"), 25, new_size)
        if video_writer.isOpened():
            for i in range(combined_video.shape[0]):
                image = np.vstack((
                    np.full((20, new_size[0]), 255),
                    cv2.resize(combined_video[i, :, :], (new_size[0], new_size[1]-20))
                )).astype(np.uint8)
                image = np.tile(image[:, :, None], [1, 1, 3])
                b = d['bestframes'][i]
                text1 = '%s'%movienames[b]
                text2 = '%d'%idframes[b]
                if keepbool[i]:
                    color = (200, 0, 0)
                else:
                    color = (100, 100, 100)

                cv2.putText(image, '%d'%i, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.CV_AA)
                cv2.putText(image, text1, (40, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.CV_AA)
                cv2.putText(image, text2, (new_size[0]-70, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.CV_AA)
                video_writer.write(image)
                if i%100 == 0:
                    print 'writing frame %d over %d\n'%(i, combined_video.shape[0])

            video_writer.release()

        plt.figure()
        plt.plot(distances)
        plt.plot(keep, distances[keep], '*g')

        # list all videos that correspond to local minima of the matching error
        chunks = []
        chunk = None
        for i in keep:
            b = d['bestframes'][i]
            name = movienames[b]
            frame = idframes[b]
            offset = i-frame+1
            if (chunk is None) or name != chunk['movie_name']:

                chunk = {'movie_name':name, 'offsets':[], 'frames':[]}
                chunks.append(chunk)


            chunk['offsets'].append(offset)
            chunk['frames'].append(i)

        with open(os.path.join(resultFolder,'timeline.txt'), 'w') as f:

            for chunk in chunks:
                start = np.min(np.array(chunk['frames']))
                end = np.max(np.array(chunk['frames']))
                offset = np.median(np.array(chunk['offsets']))
                start_source = start-offset
                text = 'From frame %d to %d : %s with an offset of %d frames (i.e from frame %d in source)\n' % (
                    start, end, chunk['movie_name'], offset, start_source
                )

                f.write(text)
        return chunks

def createThumbnails(videoFile, thumbnails_folder, reduced_size=(50, 40), basename=None, skip=10):

    video = cv2.VideoCapture(videoFile)

    idframe = 0
    nbframes = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    if not video.isOpened():
        print "cannot open video", videoFile
        input()# why?
        video = cv2.VideoCapture(videoFile)

    basename = basename or ""
    while(video.isOpened()):
        idframe += 1

        ret, frame = video.read()
        if not ret:
            print 'done'

            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #compute a descriptor
        #reduce the image to 200 pixels
        small = cv2.resize(gray, reduced_size, interpolation=cv2.INTER_AREA)

        #
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
        if idframe%skip == 0:
            print  '    frame %d over %d (%003.1f percent)'%(idframe, nbframes, 100*idframe/nbframes)
            #cv2.imshow('frame', small)
            #cv2.waitKey(1)
            framename = os.path.join(thumbnails_folder, basename + '_frame%08d.png' % idframe)
            cv2.imwrite(framename, small)


            #small2 = cv2.imread(framename)[:, :, 0]
            #print np.sum(np.abs(small2-small))

    video.release()


def loadAllSmallImageInMemory(thumbnails_folder, add_mirror=True):
    listAllSmallImages = glob.glob(os.path.join(thumbnails_folder, "*.png"))
    I = cv2.imread(listAllSmallImages[0])
    nbtotalframes = len(listAllSmallImages)
    database = np.zeros((nbtotalframes, I.shape[0], I.shape[1]), dtype=np.uint8)
    for i, image_file in enumerate(listAllSmallImages):
        database[i, :, :] = cv2.imread(image_file)[:, :, 0]
        if i%100 == 0:
            print 'loaded %d frame over %d'%(i, nbtotalframes)
    if add_mirror:
        listAllSmallImages = listAllSmallImages+listAllSmallImages
        database = np.vstack((database, database[:, :, ::-1]))
    return listAllSmallImages, database


def filter_image(image):
    image_float = image.astype(float)
    return np.abs(scipy.ndimage.convolve(image_float, [[1, -1]])) + \
        np.abs(scipy.ndimage.convolve(image_float, [[1], [-1]]))

def loadVideoSmall(videoFile, reduced_size):
    video = cv2.VideoCapture(videoFile)
    idframe = 0
    nbframes = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    videoTable = None
    while(video.isOpened()):
        idframe += 1

        ret, frame = video.read()
        if videoTable is None:
            videoTable = np.empty((nbframes, reduced_size[1], reduced_size[0]))
        if not ret:
            print 'done'

            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if idframe%(int(nbframes)/100) == 0:
            print '%d percent loaded'%(100*idframe/nbframes)
        #compute a descriptor
        #reduce the image to 200 pixels
        videoTable[idframe-1, :, :] = cv2.resize(gray, reduced_size,interpolation=cv2.INTER_AREA).astype(np.int16)
    return videoTable



def retrieveImages(videoFile, thumbnails_folder, threshold=0.1):

    listAllSmallImages, database = loadAllSmallImageInMemory(thumbnails_folder)
    reduced_size = (database.shape[2], database.shape[1])
    video = cv2.VideoCapture(videoFile)
    idframe = 0
    nbframes = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    #cv2.startWindowThread()
    #cv2.namedWindow("matching")
    plt.subplot(1, 2, 1)
    iplot = plt.imshow(np.zeros((reduced_size[0], reduced_size[1]*2)), cmap=plt.cm.Greys, vmin=0, vmax=255)
    plt.axis('off')
    plt.ion()

    plt.subplot(1, 2, 2)
    diffplot = plt.imshow(np.zeros((reduced_size[0], reduced_size[1])), vmin=-5, vmax=5)

    filtered_database = np.empty(database.shape, dtype=np.int16)
    print 'filtering database...',
    for i in range(database.shape[0]):
        filtered_database[i, :, :] = filter_image(database[i, :, :])
    print 'done'
    bestframes = np.zeros((nbframes), dtype=np.int32)
    distancestobest = np.zeros((nbframes), dtype=np.float)

    print "creating ball tree",
    BT = sklearn.neighbors.BallTree(filtered_database.reshape(filtered_database.shape[0], -1), metric='l1')
    print "done"
    start = time.clock()
    distancesMatrix = np.zeros((nbframes, filtered_database.shape[0]))
    distancetosecondbest= np.zeros((nbframes), dtype=np.float)
    while(video.isOpened()):
        idframe += 1

        ret, frame = video.read()
        if not ret:
            print 'done'

            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #compute a descriptor
        #reduce the image to 200 pixels
        small = cv2.resize(gray, reduced_size,interpolation=cv2.INTER_AREA).astype(np.int16)
        filtered_small = filter_image(small)
        ## Get nearest neighbours

        distances, bestframe = BT.query(filtered_small[None, :, :].flatten(), k=2)
        bestframe = bestframe[0, 0]
        distance = distances[0, 0]/filtered_small[None, :, :].flatten().size
        if False:#brute force
            distances = np.mean(np.abs(filtered_database-filtered_small[None, :, :]), axis=(1, 2))
            #distancesMatrix[idframe-1, :] = distances
            b = np.argmin(distances)
            bestframe = b
            distance = distances[b]
        elapsed = time.clock()-start
        fps = float(idframe)/elapsed
        predicted = (nbframes-idframe)/fps
        print 'time left %d sec , best matching image for frame %d is %s, with distance %f' % (
            predicted, idframe, listAllSmallImages[bestframe], distance
        )
        bestframes[idframe-1] = bestframe
        distancestobest[idframe-1] = distance
        distancetosecondbest[idframe-1]=distances[0, 1]/filtered_small[None, :, :].flatten().size
        np.hstack((filtered_small, filtered_database[bestframe]))
        combined = np.hstack((small, database[bestframe]))
        iplot.set_data(255-combined)
        diffplot.set_data(small-database[bestframe])
        plt.show()

    video.release()
    return bestframes, distancestobest, distancesMatrix,distancetosecondbest


def test():
    # create a rush folder
    rushesFolder='./test/rushes'  
    
    os.makedirs(rushesFolder)
    
    # create synthetic rushes
    print "creating synthetic rushes"
    for i in range(1,4):
        size=(320,240)
        videoRush=os.path.join(rushesFolder,'rush%d.avi'%i)
        video_writer = cv2.VideoWriter(videoRush, cv2.cv.FOURCC(*"MJPG"), 25, size)
        if video_writer.isOpened():
            for j in range(1,101):
                image = np.zeros((size[1],size[0],3),dtype=np.uint8)
                color = (100, 100, 100)
                cv2.putText(image,  'rush %d '%(i), (20,100), cv2.FONT_HERSHEY_SIMPLEX,1.1, color, 3, cv2.CV_AA)
                cv2.putText(image,  'frame %d'%(j), (20,130), cv2.FONT_HERSHEY_SIMPLEX,1.1, color, 3, cv2.CV_AA)
                video_writer.write(image)   
        video_writer.release()
    
    # create an edited video
    print "creating synthetic edited video"
    size=(320,240)
    videoFile=os.path.join('./test/edited.avi')
    video_writer = cv2.VideoWriter(videoFile, cv2.cv.FOURCC(*"MJPG"), 25, size)    
    for i in range(1,4):
        for j in range(1,101):
            image = np.zeros((size[1],size[0],3),dtype=np.uint8)
            color = (100, 100, 100)
            cv2.putText(image,  'rush %d '%(i), (20,100), cv2.FONT_HERSHEY_SIMPLEX,1.1, color, 3, cv2.CV_AA)
            cv2.putText(image,  'frame %d'%(j), (20,130), cv2.FONT_HERSHEY_SIMPLEX,1.1, color, 3, cv2.CV_AA)
            video_writer.write(image)
    video_writer.release()       
    # run the algorithm
    thumbnails_folder='./test/thumbnails'
    resultFolder='./test'
    chunks=main(rushesFolder,videoFile,thumbnails_folder,resultFolder)
    
    assert(len(chunks)==3)
    assert np.all(chunks[0]['frames']==np.arange(100))
    assert(chunks[0]['movie_name']=='rush1.avi')
    assert(np.all(chunks[0]['offsets']==np.zeros((100))))
    
  
    assert np.all(chunks[1]['frames']==np.arange(100,200))
    assert(chunks[1]['movie_name']=='rush2.avi')
    assert(np.all(chunks[0]['offsets']==np.zeros((100))))
    

    assert np.all(chunks[2]['frames']==np.arange(200,300))
    assert(chunks[2]['movie_name']=='rush3.avi')
    assert(np.all(chunks[0]['offsets']==np.zeros((100))))    
    
    print chunks
    shutil.rmtree(rushesFolder)

#import unittest
#testcase = unittest.FunctionTestCase(test)

if __name__ == "__main__":
    rushesFolder,videoFile,thumbnails_folder,resultFolder=getPathsInteractive()
    main(rushesFolder,videoFile,thumbnails_folder,resultFolder)
    
    #test()
