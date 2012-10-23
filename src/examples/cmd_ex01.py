python2.7 build_core_space.py -i ../examples/in/ex01 --input_format sm -o ../examples/out/ 

python2.7 build_core_space.py -i ../examples/in/ex01 --input_format sm --export_format dm
			-w ppmi,plog -r svd_200 -n none,row -o ../examples/out/ -l ../examples/out/ex01.log