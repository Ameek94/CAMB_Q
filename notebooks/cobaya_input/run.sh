#!/bin/bash

# Define the process pattern to search for
process_pattern="python spline_minimize_varyDEonly_camspec.py" # nautilus_cached_kernel_lownoise.py"

# Capture the PID of the process
pid=$(pgrep -f "$process_pattern")

if [ -n "$pid" ]; then
    echo "Process is running with PID: $pid. Waiting for it to finish..."

    # Loop until the process is no longer running
    while kill -0 "$pid" 2>/dev/null; do
        sleep 1
    done

    echo "Process with PID $pid has finished."
else
    echo "No process found matching '$process_pattern'."
fi

python spline_minimize_all_camspec.py bobyqa --maxfun 500 --nrestart 11