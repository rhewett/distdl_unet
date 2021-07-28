def mprof_wrap(routine, outfile_base=None):

    import os, subprocess

    # PID of the calling process
    my_pid = os.getpid()

    if outfile_base is None:
        outfile_base = f"mprofile_{str(my_pid)}.dat"
    outfile = f"{outfile_base}.dat"

    # Call mprof and attach it to myself
    program = ["mprof", "run", "-o", outfile, "--attach", str(my_pid)]
    profiler_pid = subprocess.Popen(program)

    # call the routine we want to profile
    routine()

    # Because we spawn the profiler, we need to kill it before the program can complete
    profiler_pid.terminate()

