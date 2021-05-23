#!/usr/bin/env bash

echo "Ranking sampling policies evaluated on all 6 learners (Table IV  individual rankings) using inputs (txt files) in ./output/table4/"

cat ./output/table4/zbrier.txt | python2 run_scott_knott.py --text 30 --latex False > ./output/table4/scratch_table4_brier.csv

echo "1 of 7 csv files generated"

cat ./output/table4/zd2h.txt | python2 run_scott_knott.py --text 30 --latex False > ./output/table4/scratch_table4_d2h.csv

echo "2 of 7 csv files generated"

cat ./output/table4/zrecall.txt | python2 run_scott_knott.py --text 30 --latex False > ./output/table4/scratch_table4_recall.csv

echo "3 of 7 csv files generated"

cat ./output/table4/zpf.txt | python2 run_scott_knott.py --text 30 --latex False > ./output/table4/scratch_table4_pf.csv

echo "4 of 7 csv files generated"

cat ./output/table4/zroc_auc.txt | python2 run_scott_knott.py --text 30 --latex False > ./output/table4/scratch_table4_auc.csv

echo "5 of 7 csv files generated"

cat ./output/table4/zifa.txt | python2 run_scott_knott.py --text 30 --latex False > ./output/table4/scratch_table4_ifa.csv

echo "6 of 7 csv files generated"

cat ./output/table4/zg-score.txt | python2 run_scott_knott.py --text 30 --latex False > ./output/table4/scratch_table4_gm.csv

echo "7 of 7 csv files generated"

