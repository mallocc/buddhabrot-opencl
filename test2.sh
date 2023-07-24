# make sure directory exists
args=()

#yes y | ./ffmpeg.exe -f image2 -framerate 24 -i output/test%d.png output/test.gif
yes y |./ffmpeg.exe -framerate 30 -i output/test%d.png -c:v libx264 -pix_fmt yuv420p output/test.mp4