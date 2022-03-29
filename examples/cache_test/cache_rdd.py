N = None
delay = 5

def addNR_dist(it):
    import threading
    import time
    result = []
    
    global N
    if N:
        print("USING CACHE!!!")
        N = N + 1.0
    else:
        print("{} rdd loading".format(threading.get_ident()))
        N = 1.0
        time.sleep(delay)
        print("{} rdd done".format(threading.get_ident()))
        
    for x in it:
        print("x: {}".format(x))
        result.append(x[1] + N)
    return result

