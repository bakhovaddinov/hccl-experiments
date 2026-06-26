while true; do
    echo "--- $(date +%s.%N) ---" >> npu_mem.log
    npu-smi info >> npu_mem.log
done
