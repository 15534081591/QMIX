if [ ! -d out ]; then
  mkdir out
fi
cd out || exit
cmake .. \
    -DMINDSPORE_PATH="`pip show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"
make
