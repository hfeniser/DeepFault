import os
import datetime
distances   = [0.2, 0.1, 0.05, 0.01]
labels      = [2,4,5,6,7,8,9]
model_names = ['mnist_test_model_5_30', 'mnist_test_model_8_20',
               'mnist_test_model_6_25']
arch        = [(5,30), (8,20), (6,25)]
percents    = [90, 95, 99]
activations = ['leaky_relu']
techniques  = ['ochiai', 'tarantula', 'random']
repeat      = [1,2,3,4,5]
suspics_num = [1,2,3,5,10]

for distance in distances:
    for label in labels:
        for model_name in model_names:
            for tech in techniques:
                for sn in suspics_num:
                    for rep in repeat:
                        if (not tech == 'random') and rep > 1:
                            continue

                        command = 'python run.py -A ' + tech + ' -N ' + \
                        str(arch[model_names.index(model_name)][1]) + ' -HL ' +\
                        str(arch[model_names.index(model_name)][0]) + ' -C ' +\
                        str(label) + ' -AC leaky_relu -D ' +\
                        str(distance) + ' -M ' + \
                        model_name + ' -R ' + str(rep) + ' -S ' +str(sn) + \
                        ' -LOG experiment/logfile.log'

                        start = datetime.datetime.now()
                        os.system(command)
                        end = datetime.datetime.now()
                        logfile = open('experiment/logfile.log','a')
                        logfile.write('Total time (including preperations): ' + str(end-start))
                        logfile.close()
