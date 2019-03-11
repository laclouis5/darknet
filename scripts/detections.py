import os
import subprocess

result      = 'result.txt'
save_folder = 'detections/'
val_dir     = '../data/val/'
classes     = ['mais', 'haricot', 'carotte']
gt_folder   = 'groundtruths/'

height, width = 2048, 2448

with open(result, 'r') as f:
    output    = []
    name_list = []
    lines     = f.readlines()
    indices   = [i for (i, line) in enumerate(lines) if 'Image Path:' in line]

    for i, start in enumerate(indices[:-1]):
        stop = indices[i+1]
        temp = []

        first_line = lines[start]
        fields     = first_line.split(' ')
        name       = os.path.splitext(os.path.split(fields[3][:-1])[1])[0] + '.txt'

        name_list.append(name)

        for line in lines[start+1:stop]:
            # Find class name and confidence
            components = line.split('\t')
            for item in classes:
                if item in components[0]:
                    label = item

            i_colon = components[0].find(':')
            i_percent = components[0].find('%')
            confidence = float(components[0][i_colon+2:i_percent]) / 100

            # find bbox information
            i_x = components[1].find('left_x')
            i_y = components[1].find('top_y')
            i_w = components[1].find('width')
            i_h = components[1].find('height')

            x = int(components[1][i_x+8:i_y])
            y = int(components[1][i_y+7:i_w])
            w = int(components[1][i_w+7:i_h])
            h = int(components[1][i_h+8:len(components[1])-2])

            line = '{} {} {} {} {} {}\n'.format(
                label,
                confidence,
                x,
                y,
                w,
                h
            )
            temp.append(line)
        output.append(temp)


    for i, name in enumerate(name_list):
        with open(os.path.join(save_folder, name), 'w') as f_write:
            f_write.writelines(output[i])

list = os.listdir(val_dir)
list = [item for item in list if os.path.splitext(item)[1] == '.txt']
dic  = {0: 'mais', 1: 'haricot', 2: 'carotte'}

for item in list:
    temp = []
    with open(os.path.join(val_dir, item), 'r') as f:
        lines = f.readlines()
        lines = [line.strip('\n') for line in lines]
        lines = [line.split(' ') for line in lines]

        for line in lines:
            buf = [float(element) for element in line[1:]]

            buf[0] = int((buf[0] - buf[2] / 2) * width)
            buf[1] = int((buf[1] - buf[3] / 2) * height)
            buf[2] = int(buf[2] * width)
            buf[3] = int(buf[3] * height)

            new_line  = '{} {} {} {} {}\n'.format(dic[int(line[0])], buf[0], buf[1], buf[2], buf[3])
            temp.append(new_line)

    with open(os.path.join(gt_folder, item), 'w') as f_write:
        f_write.writelines(temp)

        # buf = [element for line in lines for element in line]
        # print(buf)
