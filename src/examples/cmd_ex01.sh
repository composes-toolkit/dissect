python2.7 build_core_space.py -i ../examples/data/in/ex01 --input_format sm -o ../examples/data/out/
python2.7 build_core_space.py -i ../examples/data/in/ex01 --input_format sm --output_format dm -w ppmi,plog -r svd_2 -n none,row -o ../examples/data/out/ -l ../examples/data/out/ex01.log
#or
python2.7 build_core_space.py ../examples/data/in/config1.cfg
python2.7 build_core_space.py ../examples/data/in/config2.cfg
