python generator.py --code=surface --basis=z --noise=circuit -d 7 -p 0.0005 --records=10000000 &
python generator.py --code=surface --basis=z --noise=circuit -d 7 -p 0.001 --records=10000000 &
python generator.py --code=surface --basis=z --noise=circuit -d 7 -p 0.003 --records=10000000 & 
python generator.py --code=surface --basis=z --noise=circuit -d 7 -p 0.005 --records=10000000 &
wait
python generator.py --code=qldpc --basis=z --noise=circuit -d 6 -p 0.0005 --records=10000000 &
python generator.py --code=qldpc --basis=z --noise=circuit -d 6 -p 0.001 --records=10000000 &
python generator.py --code=qldpc --basis=z --noise=circuit -d 6 -p 0.003 --records=10000000 &
python generator.py --code=qldpc --basis=z --noise=circuit -d 6 -p 0.005 --records=10000000 &
wait
python generator.py --code=qldpc --basis=z --noise=circuit -d 12 -p 0.0005 --records=100000000
python generator.py --code=qldpc --basis=z --noise=circuit -d 12 -p 0.001 --records=100000000
python generator.py --code=qldpc --basis=z --noise=circuit -d 12 -p 0.003 --records=100000000
python generator.py --code=qldpc --basis=z --noise=circuit -d 12 -p 0.005 --records=100000000
python generator.py --code=color --basis=z --noise=circuit -d 7 -p 0.0005 --records=10000000 &
python generator.py --code=color --basis=z --noise=circuit -d 7 -p 0.001 --records=10000000 &
python generator.py --code=color --basis=z --noise=circuit -d 7 -p 0.003 --records=10000000 &
python generator.py --code=color --basis=z --noise=circuit -d 7 -p 0.005 --records=10000000 &
wait

python generator.py --code=ls --basis=z --noise=circuit -d 11 -p 0.0005 --records=20000000
python generator.py --code=ls --basis=z --noise=circuit -d 11 -p 0.001 --records=20000000
python generator.py --code=ls --basis=z --noise=circuit -d 11 -p 0.003 --records=20000000
python generator.py --code=ls --basis=z --noise=circuit -d 11 -p 0.005 --records=20000000
