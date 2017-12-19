import os
import time
import glob
import re
import io
import fileinput
import psutil

PID = 0

def getTasks(name):
    r = os.popen('tasklist /v').read().strip().split('\n')
    print('# of tasks is %s' % (len(r)))
    for i in range(len(r)):
        s = r[i]
        temp = s[:-175]
        if name in r[i]:
            print('%s in r[i]' % (name))
            PID = int(re.search(r'\d+', temp).group())
            return r[i]
    return []



if __name__ == '__main__':
    weight_folder = "D://Users//208018409//Documents//darknet-master//build//darknet//x64//backup"

    while (1):
        '''
        This code checks tasklist, and will print the status of a code
        '''

        imgName = 'darknet.exe'

        notResponding = 'Not Responding'

        r = getTasks(imgName)


        if not r:
            print('%s - No such process' % (imgName))
            #Read stored weights
            os.chdir(weight_folder)
            file = glob.glob("*.weights")

            #loop through files to find that lastest
            weight_number = 0
            for m in range (0,len(file)):
                print(file[m])
                temp = int(re.search(r'\d+', file[m]).group())
                if (temp > weight_number):
                    weight_number = temp
            #weight_number = int(re.search(r'\d+', latest_weight_file).group())

            os.chdir("D:/Users/208018409/Documents/darknet-master/build/darknet/x64/")

            #replace steps in config file
            f = open("yolo-voc.2.0_fish.cfg", 'r')  # pass an appropriate path of the required file
            lines = f.readlines()
            n = 17
            lines[n - 1] = "steps=" + str(weight_number + 100) + "\n" # n is the line number you want to edit; subtract 1 as indexing of list starts from 0
            f.close()  # close the file and reopen in write mode to enable writing to file; you can also open in append mode and use "seek", but you will have some unwanted old data if the new data is shorter in length.

            f = open("yolo-voc.2.0_fish.cfg", 'w')
            f.writelines(lines)
            # do the remaining operations on the file
            f.close()

            if (weight_number == 0):
                command = "start cmd /c darknet.exe detector train data/fish2.data yolo-voc.2.0_fish.cfg darknet19_448.conv.23"
            else:
                command = "start cmd /c darknet.exe detector train data/fish2.data yolo-voc.2.0_fish.cfg" + " " + "backup/yolo-voc_" + str(weight_number) + ".weights"

            os.system(command)
            time.sleep(20)
        elif 'Not Responding' in r:
            print('%s is Not responding' % (imgName))
        else:
            print('%s is Running or Unknown' % (imgName))
            p = psutil.Process(PID)
            cpu = p.cpu_percent(interval=1)
            print(cpu)
            if (cpu > 1100):
                os.system("taskkill /f /im darknet.exe")
                os.system("taskkill /f /im WerFault.exe")
    time.sleep(10)