import multiprocessing
import time
def f(x):
    print('started nr {}'.format(x))
    #Programmet stopper i 1 sekund
    time.sleep(1)
    return x*x

def go():
    start = time.time()
    #'processes' angir ntall prosessorer
    pool = multiprocessing.Pool(processes=4)
    print(pool.map(f, range(10)))
    end = time.time()
    print('finished in {0:4.3f}'.format(end-start))

#Følgende må stå i mainfil
#if __name__== '__main__' :
#    multiprocessing.freeze_support()
#    go()