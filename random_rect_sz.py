import matplotlib.pyplot as plt
import numpy as np
import sys
import pymongo
import os

from scipy import stats

data_file_name = ".data"

def ping_mongo():
    hostname = "192.168.0.101"
    response = os.system("ping -c 1 " + hostname)
    return False if response else True

def get_query():
    if ping_mongo():
        myclient = pymongo.MongoClient("mongodb://192.168.0.101:27017/")
        mydb = myclient["label_clicker"]
        mycol = mydb["train_taskimages_clean"]
        myquery = {"data.skipped": False, "data.canvas_objects.0":{"$exists": True}}   
        mydata = mycol.find(myquery)
        print('Reading from mongoDB...') 
        return(mydata)
    else:       
        print('Reading from file...')
        return(0)


def get_rect_data():       
    sample = get_query()   
    l = []

    if sample == 0:       
        f = open(data_file_name, 'r')
        for line in f:
            l.append(float(line))
        f.close()
    else:
        os.remove(data_file_name)
        f = open(data_file_name, 'w+')
        for x in sample:
            canvas_objects = x['data'][0]['canvas_objects']
            for y in canvas_objects:
                h = y['h']; w = y['w']
                size = 0.75 * max(w, h) + 0.25 * min(w, h)
                l.append(size)
                f.write(str(size) + '\n')
        f.close()   
    # creating data
    l_list = list(np.histogram(l, bins=30))
    min_x = min(l_list[1])
    max_x = max(l_list[1])
    shape, loc, scale = stats.lognorm.fit(l, floc=0)
    # print data
    print('='*28, '\nRectangles count:', len(l))
    print('Min(x):', min_x, '\nMax(x):', max_x)
    print('Shape:', shape, '\nLoc:', loc, '\nScale:', scale)
    print('='*28,'\nCreating a histogram...')
    # return data
    return([l, min_x, max_x, len(l), shape, loc, scale])


def get_random_rect_sz_HARDCODE():
    shape, scale = 0.7635779560378387, 0.07776496289182451
    dist = stats.lognorm(shape, 0.0, scale)
    size = dist.rvs()
    size = size * 0.9671784150570207 + 0.007142151004612083
    return(size)


def get_random_rect_sz(min_x, max_x, shape, loc, scale):
    dist = stats.lognorm(shape, loc, scale)
    size = dist.rvs()
    size = size * (max_x - min_x) + min_x
    return(size)
            

if __name__ == "__main__":
    random_rect_sz_list = []
    # get data
    rect_sz_list, min_x, max_x, rects_count, shape, loc, scale = get_rect_data()  
    # create random sizes
    for i in range(rects_count):
        random_rect_sz_list.append(get_random_rect_sz(min_x, max_x, shape, loc, scale))
    # histogram show
    plt.hist(rect_sz_list, bins=30, alpha=0.5, label='Brands')
    plt.hist(random_rect_sz_list, bins=30, alpha=0.5, label='Random')
    plt.legend(loc='upper right') 
    plt.show()
    print('Done!')