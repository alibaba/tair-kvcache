if [ -z $1 ]; then
    echo 'need <$tid> [$dst_name.svg]'
    exit 1
fi

now=$(date +%s)
set -e
set -x
sudo perf record -e cpu-clock --call-graph dwarf -t $1 -o perf.$now.data -- sleep 20
sudo perf script -i perf.$now.data > perf.$now.script
sudo ad cmd flamegraph stackcollapse-perf.pl perf.$now.script  > perf.$now.folded
sudo ad cmd flamegraph flamegraph.pl perf.$now.folded > perf.$now.svg

if [ -z $2 ]; then
    sudo ad put perf.$now.svg
else
    sudo ad put perf.$now.svg $2
fi
