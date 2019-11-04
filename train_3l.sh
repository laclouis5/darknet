for INDEX in 1 2 3 4 5 6 7 8 9 10
do
    ./darknet detector train data/obj.data cfg/yolov3-tiny_3l.cfg -map
    FOLDER="results/yolov3-tiny_3l_test_"$INDEX
    mkdir $FOLDER
    mv chart.png $FOLDER
    mv backup/yolov3-tiny_3l_best.weights $FOLDER  
done
