# make sure directory exists
args=()

args+=" -steps 60"
#args+=" -nc -x0 -0.970833333333333333235 -x1 -0.608333333333333333206 -y0 0.066666666666666666552 -y1 0.429166666666666666581"
#args+=" -x0 0.300000000000000000200 -y0 0.595833333333333333261 -x1 0.416666666666666666876 -y1 0.712499999999999999937"
#args+=" -ir 256 -ig 1024 -ib 4096 -im 128"
#args+=" -nc -zsc 0.2 -zysc -2 "
args+=" -nc -x0 -1.454166666666666666607 -y0 -0.112500000000000000129 -x1 -1.216666666666666666588 -y1 0.112500000000000000129"
args+=" -nc -x0 -2 -y0 -1.5 -x1 1 -y1 1.5"
args+=" -nc -x0 -1.316666666666666666596 -y0 0.145833333333333333225 -x1 -1.016666666666666666572 -y1 0.445833333333333333249"
#args+=" -nc -a 90 -zsc 0.619 -zac 134 -zysc -0.657 "
#args+=" -nc -a 180 -b 90 -zsb 0.160 -zab -16 -zysb 0.485"
#args+=" -nc -a 240 -b 180 -t 90 -zsa 0.000 -zaa 130 -zysa 0.356 "
#args+=" -nc -a 360 -b 240 -t 180 -p 90 -zsc 0.619 -zac 134 -zysc -0.657 "
#args+=" -n "

#./x64/Release/buddhabrot-opencl.exe -bezier-enable -gamma 2.5 -w 1280 -h 720 -s 10000000 -ir 2000 -ig 500 -ib 100 -im 50 -o output/test_ ${args[*]}
#./x64/Release/buddhabrot-opencl.exe -silent -bezier-enable -gen-in-region -w 256 -h 256 -s 1000000 -ir 1000 -ig 250 -ib 100 -im 50 -o output/test1_ -steps 30 -x0 -1.37422 -y0 -0.0863194 -x1 -1.37176 -y1 -0.0838629 -nc -b 10 -a 60 -nc -a 89 -b 89 -nc -b 90 -a 90 -nc -a 90.5 -b 90.5 -steps 120 -nc -x0 -2 -y0 -1.5 -x1 1 -y1 1.5 -a 0 -b 180 -steps 60 -nc -b 0 -steps 30 -nc -steps 90 -nc -x0 -1.37422 -y0 -0.0863194 -x1 -1.37176 -y1 -0.0838629
./x64/Release/buddhabrot-opencl.exe -gen-in-region -gamma 2.5 -w 256 -h 256 -s 1000000 -ir 1024 -ig 256 -ib 128 -im 64 -x0 -2 -y0 -1.5 -x1 1 -y1 1.5 -o output/test ${args[*]}

#yes y | ./ffmpeg.exe -f image2 -framerate 30 -i output/test%d.png output/test.gif
yes y |./ffmpeg.exe -framerate 30 -i output/test%d.png -c:v libx264 -pix_fmt yuv420p output/test.mp4