import os
list1=os.listdir('traj')
for i in range(len(list1)):
    os.system('python main.py' + ' ' + str(i))