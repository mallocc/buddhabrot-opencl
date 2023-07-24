# make sure directory exists
args=()

#args+="-steps 120 "
#args+=" -nc -a 90 -zsc 0.619 -zac 134 -zysc -0.657 "
#args+=" -nc -a 180 -b 90 -zsb 0.160 -zab -16 -zysb 0.485"
#args+=" -nc -a 240 -b 180 -t 90 -zsa 0.000 -zaa 130 -zysa 0.356 "
#args+=" -nc -a 360 -b 240 -t 180 -p 90 -zsc 0.619 -zac 134 -zysc -0.657 "
#args+=" -n "

./x64/Release/OpenCLTest.exe -bezier-enable -gamma 2.5 -w 1280 -h 720 -s 10000000 -ir 2000 -ig 500 -ib 100 -im 50 -o output/test_ ${args[*]}

#yes y | ./ffmpeg.exe -f image2 -framerate 24 -i output/test%d.png output/test.gif
yes y |./ffmpeg.exe -framerate 30 -i output/test_%d.png -c:v libx264 -pix_fmt yuv420p output/test_.mp4