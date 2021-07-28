from mprof import *

def plot_file(filename, index=0, timestamps=True, children=True, options=None):
    try:
        import pylab as pl
    except ImportError as e:
        print("matplotlib is needed for plotting.")
        print(e)
        sys.exit(1)
    import numpy as np  # pylab requires numpy anyway
    mprofile = read_mprofile_file(filename)

    if len(mprofile['timestamp']) == 0:
        print('** No memory usage values have been found in the profile '
              'file.**\nFile path: {0}\n'
              'File may be empty or invalid.\n'
              'It can be deleted with "mprof rm {0}"'.format(
            mprofile['filename']))
        sys.exit(0)

    # Merge function timestamps and memory usage together
    ts = mprofile['func_timestamp']
    t = mprofile['timestamp']
    mem = mprofile['mem_usage']
    chld = mprofile['children']

    if len(ts) > 0:
        for values in ts.values():
            for v in values:
                t.extend(v[:2])
                mem.extend(v[2:4])

    mem = np.asarray(mem)
    t = np.asarray(t)
    ind = t.argsort()
    mem = mem[ind]
    t = t[ind]

    # Plot curves
    global_start = float(t[0])
    t = t - global_start

    max_mem = mem.max()
    max_mem_ind = mem.argmax()

    all_colors = ("c", "y", "g", "r", "b")
    mem_line_colors = ("k", "b", "r", "g", "c", "y", "m")

    show_trend_slope = options is not None and hasattr(options, 'slope') and options.slope is True

    mem_line_label = time.strftime("%d / %m / %Y - start at %H:%M:%S",
                                   time.localtime(global_start)) \
                     + ".{0:03d}".format(int(round(math.modf(global_start)[0] * 1000)))

    mem_trend = None
    if show_trend_slope:
        # Compute trend line
        mem_trend = np.polyfit(t, mem, 1)

        # Append slope to label
        mem_line_label = mem_line_label + " slope {0:.5f}".format(mem_trend[0])

    pl.plot(t, mem, "+-" + mem_line_colors[index % len(mem_line_colors)],
            label=filename)

    if show_trend_slope:
        # Plot the trend line
        pl.plot(t, t*mem_trend[0] + mem_trend[1], "--", linewidth=0.5, color="#00e3d8")

    bottom, top = pl.ylim()
    bottom += 0.001
    top -= 0.001

    # plot children, if any
    if len(chld) > 0 and children:
        cmpoint = (0,0) # maximal child memory

        for idx, (proc, data) in enumerate(chld.items()):
            # Create the numpy arrays from the series data
            cts  = np.asarray([item[1] for item in data]) - global_start
            cmem = np.asarray([item[0] for item in data])

            cmem_trend = None
            child_mem_trend_label = ""
            if show_trend_slope:
                # Compute trend line
                cmem_trend = np.polyfit(cts, cmem, 1)

                child_mem_trend_label = " slope {0:.5f}".format(cmem_trend[0])

            # Plot the line to the figure
            pl.plot(cts, cmem, "+-" + mem_line_colors[(idx + 1) % len(mem_line_colors)],
                    label="child {}{}".format(proc, child_mem_trend_label))

            if show_trend_slope:
                # Plot the trend line
                pl.plot(cts, cts*cmem_trend[0] + cmem_trend[1], "--", linewidth=0.5, color="black")

            # Detect the maximal child memory point
            cmax_mem = cmem.max()
            if cmax_mem > cmpoint[1]:
                cmpoint = (cts[cmem.argmax()], cmax_mem)

        # Add the marker lines for the maximal child memory usage
        pl.vlines(cmpoint[0], pl.ylim()[0]+0.001, pl.ylim()[1] - 0.001, 'r', '--')
        pl.hlines(cmpoint[1], pl.xlim()[0]+0.001, pl.xlim()[1] - 0.001, 'r', '--')

    # plot timestamps, if any
    if len(ts) > 0 and timestamps:
        func_num = 0
        f_labels = function_labels(ts.keys())
        for f, exec_ts in ts.items():
            for execution in exec_ts:
                add_brackets(execution[:2], execution[2:], xshift=global_start,
                             color=all_colors[func_num % len(all_colors)],
                             label=f_labels[f]
                                   + " %.3fs" % (execution[1] - execution[0]), options=options)
            func_num += 1

    if timestamps:
        pl.hlines(max_mem,
                  pl.xlim()[0] + 0.001, pl.xlim()[1] - 0.001,
                  colors="r", linestyles="--")
        pl.vlines(t[max_mem_ind], bottom, top,
                  colors="r", linestyles="--")
    return mprofile