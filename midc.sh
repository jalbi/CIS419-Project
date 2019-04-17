CURR=0
for i in $(ls data); do
    ./midicsv data/$i datac/$CURR.csv
    CURR=$((CURR+1))
done
