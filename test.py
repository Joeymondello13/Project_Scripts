import time
start_time = time.time()
for i in range(100000):
    print(i)
time_file = open('time.txt','a')
time_file.writelines('trajname: ' + str(round((time.time() - start_time)/60,2)) +'\n')
time_file.close()