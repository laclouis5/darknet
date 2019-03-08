import os
import subprocess

eval_folder = "backup/"
cfg_file    = "cfg/yolov3-tiny_obj.cfg"
data_file   = "data/obj.data"
labels      = {0: 'mais', 1: 'haricot', 2: 'carotte'}

cwd = '/home/deepwater/github/darknet/'

networks = sorted(os.listdir(os.path.join('../', eval_folder)))
nb_net   = len(networks)

# result = {network, mAP, mAP}
results = []

for network in networks:
    network_path = os.path.join(eval_folder, network)
    command = "./darknet detector map {} {} {}".format(data_file, cfg_file, network_path)

    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, cwd=cwd)
    output, error = process.communicate()

    output = output.decode('utf-8')
    output = output.splitlines()
    output = [str for str in output if '%' in str]

    mAPs = []
    for string in output:
        new_string = string[-8:-3]
        if '=' not in new_string:
            mAPs.append(float(new_string) / 100)

    results.append(mAPs)

nb_networks = len(results)
nb_classes  = len(results[0])-2

with open('log.csv', 'w') as f:
    f.write('iteration,mAP')

    for i in range(nb_classes):
        f.write(',{}'.format(labels[i]))
    f.write(',IoU\n')

    for j, network in enumerate(networks):
        data = results[j]
        idx = network.find('.')
        string1 = network[idx-4:idx]
        string2 = network[idx-3:idx]
        if string1.isdigit():
            f.write('{},{}'.format(int(string1), data[-1]))
            for k in range(nb_classes):
                f.write(',{}'.format(data[k]))
            f.write(',{}\n'.format(data[nb_classes]))
        elif string2.isdigit():
            f.write('{},{}'.format(int(string2), data[-1]))
            for k in range(nb_classes):
                f.write(',{}'.format(data[k]))
            f.write(',{}\n'.format(data[nb_classes]))
