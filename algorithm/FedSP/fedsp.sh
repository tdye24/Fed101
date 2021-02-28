echo "nohup python fedsp-main.py -dataset ${1} -model ${2} --lr ${3} --lr-decay ${4} --batch-size ${5} --clients-per-round ${6} --num-rounds ${7} --seed ${8} --epoch ${9} --eval-interval ${10} --note ${11} > ${1}_${2}_C${6}_E${9}_B${5}_lr${3}_${11}.txt 2>&1 &"
nohup python fedsp-main.py -dataset ${1} -model ${2} --lr ${3} --lr-decay ${4} --batch-size ${5} --clients-per-round ${6} --num-rounds ${7} --seed ${8} --epoch ${9} --eval-interval ${10} --note ${11} > ${1}_${2}_C${6}_E${9}_B${5}_lr${3}_${11}.txt 2>&1 &

