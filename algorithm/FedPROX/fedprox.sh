echo "nohup python fedprox-main.py -dataset ${1} -model ${2} --lr ${3} --lr-decay ${4} --decay-step ${5} --batch-size ${6} --clients-per-round ${7} --num-rounds ${8} --seed ${9} --epoch ${10} --eval-interval ${11} --note ${12} > ${1}_${2}_C${7}_E${10}_B${6}_lr${3}_${12}.txt 2>&1 &"
nohup python fedprox-main.py -dataset ${1} -model ${2} --lr ${3} --lr-decay ${4} --decay-step ${5} --batch-size ${6} --clients-per-round ${7} --num-rounds ${8} --seed ${9} --epoch ${10} --eval-interval ${11} --note ${12} > ${1}_${2}_C${7}_E${10}_B${6}_lr${3}_${12}.txt 2>&1 &

