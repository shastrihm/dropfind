@ECHO OFF
 
start /MIN python dropfind.py -p "test_install\test" -n 20 -b "J000597" -m True 
python dropfind_tests.py -t "basic"

timeout 10 1>NUL

start /MIN python dropfind.py -p "test_install\test" -n 20 -b "J000597" -m True 
python dropfind_tests.py -t "longrun" -i 1.6

timeout 10 1>NUL

start /MIN python dropfind.py -p "test_install\test" -n 20 -b "J000597" -m True 
python dropfind_tests.py -t "longrun" -i 0.6

timeout 10 1>NUL

start /MIN python dropfind.py -p "test_install\test" -n 20 -b "J000597" -m True 
python dropfind_tests.py -t "stop" -i 1

timeout 10 1>NUL

start /MIN python dropfind.py -p "test_install\test" -n 20 -b "J000597" -m True 
python dropfind_tests.py -t "rollover"

