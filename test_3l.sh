for INDEX in 1 2 3 4 5 6 7 8 9 10
do
    FOLDER="results/yolov3-tiny_3l_test_"$INDEX"/"
    ./darknet detector map data/obj.data cfg/yolov3-tiny_3l.cfg $FOLDER"yolov3-tiny_3l_best.weights"
done
