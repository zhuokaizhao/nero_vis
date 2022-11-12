#!/bin/bash
# the script executes essential steps for qiv to work in nero_vis/qt_app
if [ $# -eq 0 ];
then
    echo "$0: The following arguments are required: teem_install_path, teem_python_path"
    exit 1
elif [ $# -eq 1 ];
then
    echo "$0: The following argument are required: teem_python"
    exit 1
elif [ $# -gt 2 ];
then
    echo "$0: Too many arguments: $@"
    exit 1
else
    teem_install_path=$1
    teem_python_path=$2
    echo "==========================="
    echo "teem_install_path.: $teem_install_path"
    echo "teem_python_path..: $teem_python_path"
    echo "==========================="
fi

# build in nero_vis/qiv
cd ../qiv
make clean
make
python build_qiv.py $teem_install_path $teem_python_path
cd ../qt_app
