#!/bin/bash

mkdir -p ./results

commands_file="sim_commands.txt.pkl"
> "$commands_file"

add_cmd() {
    echo "python rare_event_sim.py $@" >> "$commands_file"
}

# Surface Code (d=19, 23, 27)
# Decoders: pymatching, mwpf, relay-bp
for d in 19 23 27; do
    for decoder in pymatching mwpf relay-bp; do
        add_cmd --code surface --basis z --distance "$d" --decoder "$decoder" --num_trials_per_weight 2000
    done
done

# QLDPC Code (d=12, 18)
# Decoders: bplsd, mwpf, relay-bp
for d in 12 18; do
    for decoder in bplsd mwpf relay-bp; do
        add_cmd --code qldpc --basis z --distance "$d" --decoder "$decoder" --num_trials_per_weight 2000
    done
done

# Color Code (d=19, 23, 27)
# Decoders: bplsd, mwpf, relay-bp
for d in 19 23 27; do
    for decoder in bplsd mwpf relay-bp; do
        add_cmd --code color --basis z --distance "$d" --decoder "$decoder" --num_trials_per_weight 2000
    done
done

# Lattice Surgery (d=19, 23, 27)
# Decoders: pymatching, mwpf, relay-bp
for d in 19 23 27; do
    for decoder in pymatching mwpf relay-bp; do
        add_cmd --code ls --basis z --distance "$d" --decoder "$decoder" --num_trials_per_weight 2000
    done
done

# Run with GNU parallel
if command -v parallel >/dev/null 2>&1; then
    cat "$commands_file" | parallel --bar --jobs 24
else
    echo "GNU parallel not found. Running sequentially..."
    bash "$commands_file"
fi

echo "Simulations completed."