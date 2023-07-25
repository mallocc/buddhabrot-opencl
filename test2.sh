# make sure directory exists
args=()

args+=" -zac -180 -steps 180 -nc -zsc 0.2 -zac 180 -zysc -2 "
#args+=" -nc -a 90 -zsc 0.619 -zac 134 -zysc -0.657 "
#args+=" -nc -a 180 -b 90 -zsb 0.160 -zab -16 -zysb 0.485"
#args+=" -nc -a 240 -b 180 -t 90 -zsa 0.000 -zaa 130 -zysa 0.356 "
#args+=" -nc -a 360 -b 240 -t 180 -p 90 -zsc 0.619 -zac 134 -zysc -0.657 "
#args+=" -n "

#./x64/Release/buddhabrot-opencl.exe -bezier-enable -gamma 2.5 -w 1280 -h 720 -s 10000000 -ir 2000 -ig 500 -ib 100 -im 50 -o output/test_ ${args[*]}
#./x64/Release/buddhabrot-opencl.exe -silent -bezier-enable -gen-in-region -w 256 -h 256 -s 1000000 -ir 1000 -ig 250 -ib 100 -im 50 -o output/test1_ -steps 30 -x0 -1.37422 -y0 -0.0863194 -x1 -1.37176 -y1 -0.0838629 -nc -b 10 -a 60 -nc -a 89 -b 89 -nc -b 90 -a 90 -nc -a 90.5 -b 90.5 -steps 120 -nc -x0 -2 -y0 -1.5 -x1 1 -y1 1.5 -a 0 -b 180 -steps 60 -nc -b 0 -steps 30 -nc -steps 90 -nc -x0 -1.37422 -y0 -0.0863194 -x1 -1.37176 -y1 -0.0838629
./x64/Release/buddhabrot-opencl.exe -bezier-enable -gamma 2.5 -w 512 -h 512 -s 100000000 -ir 1000 -ig 250 -ib 100 -im 50 -x0 -2 -y0 -1.5 -x1 1 -y1 1.5 -o output/test ${args[*]}

yes y | ./ffmpeg.exe -f image2 -framerate 30 -i output/test%d.png output/test.gif
#yes y |./ffmpeg.exe -framerate 30 -i output/test_%d.png -c:v libx264 -pix_fmt yuv420p output/test_.mp4