import argparse
import xml.etree.ElementTree
from collections import OrderedDict
from collections import deque
from itertools import islice
import pickle
import numpy as np

def read_xml(e, dct):
    filename=e.find('filename').text.replace('.xml','.jpg')
    #filename=filename.replace('gopr0', 'GOPR0')
    #filename=filename.replace('frame', '_')
    dct[filename] =[]
    for atype in e.findall('object'):

        nrows=float(e.find('imagesize').find('nrows').text)
        ncols=float(e.find('imagesize').find('ncols').text)

        #print nrows, ncols
        #if atype.find('name').text == 'car':
        label = [1.0]
        #elif atype.find('name').text == 'truck' or atype.find('name').text =='bus':
        #    label = [0.0, 1.0, 0.0]
        for polygon in atype.findall('polygon'):
            x,y=[], []
            for pt in polygon.findall('pt'):
                x.append(int(pt.find('x').text))
                y.append(int(pt.find('y').text))

            y=list(map(lambda x: x/nrows, y))
            x=list(map(lambda x: x/ncols, x))

            loc= [min(x), min(y), max(x), max(y)]
            loc.extend(label)
            dct[filename].append(loc)

    dct[filename] = np.asarray(dct[filename])

def create_dct(store,path = '0251',n_frames=67,video='gopr0251'):
    """
    function to read the xml file annotations
    store: dictionary element where you want to save the annotations
    path: directory of annotations
    n_frames: number of frames
    video: name of video
    """

    for i in range(n_frames):
        print(path + '/'+video+'frame'+str(i).zfill(5))
        e = xml.etree.ElementTree.parse(path + '/'+video+'frame'+str(i).zfill(5)+'.xml').getroot()

        read_xml(e, store)


if __name__ == "__main__":
    """
    usage python read_xml.py --dir './0251' -v 'gopr0251' -n 67 --save './save.pkl'
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, dest='load_dir',
                        help='Loading Dictionary Path')

    parser.add_argument('-v',type=str, dest='video_name',
                        help='Name of the video')
    parser.add_argument('-n', 
                        dest='n_frames', type=int)
    parser.add_argument('--save', type=str, 
                        dest='save_file',
                        help='Location of the pkl file to save')

  
    results = parser.parse_args()
    path = results.load_dir
    video=results.video_name
    n_frames=results.n_frames
    save_file=results.save_file
    


    store=OrderedDict()
    create_dct(store, path , n_frames, video)

    with open(save_file, "wb") as f:
        pickle.dump(store, f)
