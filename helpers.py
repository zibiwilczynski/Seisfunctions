# helpers.py
"""
Created on Wed Jan 14 11:20:53 2026

@author: zibiwilczynski
"""

from scipy.signal import convolve2d
import scipy.signal as ss
import inspect
from pathlib import Path
import logging
import gc
import glob
import os
import os.path
import h5py
import xdas as xd
import xdas.fft as xfft
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, correlation_lags, correlate
from scipy.ndimage import gaussian_filter
import matplotlib
import pandas as pd
from matplotlib.collections import PolyCollection
matplotlib.use('Agg')


def start_logging(log_dir, cfgpath, newlog=False):
    log_dir.mkdir(parents=True, exist_ok=True)
    logpath = Path(log_dir, Path(cfgpath).stem+'.log')
    logger = logging.getLogger('Production')
    logger.setLevel(logging.DEBUG)
    # IMPORTANT: remove any existing handlers (Jupyter / reruns)
    logger.handlers.clear()
    formatter = logging.Formatter('%(message)s')
    # File handler
    if newlog and logpath.is_file():
        with open(logpath, 'w'):
            pass
    fh = logging.FileHandler(logpath, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    logger.info(f"New logging run: {datetime.now()}")
    return logger


def debug(msg):
    logger = logging.getLogger('Production')
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(
        inspect.currentframe())[1]
    line = lines[0] if lines else ""  # ← safe fallback
    indentation_level = line.find(line.lstrip())
    logger.info('{i} {m}'.format(
        i='.'*indentation_level,
        m=msg
    ))


def scan_dir_for_files(root_dir, extension):
    """
    Recursively find all files under root_dir with the given extension.
    extension: e.g. '.hdf5', '.nc', '.mseed'
    Returns: list of absolute file paths.
    """
    matches = glob.glob(os.path.join(
        root_dir, "**", f"*{extension}"), recursive=True)
    debug(f"Found {len(matches)} files in the directory")
    debug(f"First file: {os.path.basename(matches[0])}")
    debug(f"Last file:  {os.path.basename(matches[-1])}")
    return matches


def explore_h5_metadata(path):
    """
    Recursively print all groups/datasets and their attributes in an HDF5 file.
    """
    debug('Exploring files metadata')
    # Handle both single file and list
    if isinstance(path, str):
        pass
    elif isinstance(path, (list, tuple)):
        path = path[0]
    elif not isinstance(path, (str, list, tuple)):
        raise ValueError("files must be str or list/tuple of str")
    with h5py.File(path, "r") as f:
        def _print_attrs(name, obj):
            # debug(f"\n{name} ({type(obj).__name__})")
            # File / group / dataset attributes
            for key, val in obj.attrs.items():
                debug(f"  {key}: {val!r}")
        f.visititems(_print_attrs)


def read_file_list(files, tolerance=100, overlap=None, off=None):
    """
    Read all the files in a file list. Tolerance in [ms].
    """
    debug("Loading files sequentially...")
    das = []

    # Handle both single file and list
    if isinstance(files, str):
        files = [files]
    elif not isinstance(files, (list, tuple)):
        raise ValueError("files must be str or list/tuple of str")

    for i, f in enumerate(files, 1):
        debug(f"File {i}/{len(files)}: {os.path.basename(f)}")
        da = xd.open_dataarray(f, engine='febus', overlaps=(overlap[0], overlap[1]), offset=off)
        das.append(da)

    debug("Combining...")
    da_sequence = xd.combine_by_coords(
        das,
        dim='time',
        tolerance=np.timedelta64(tolerance, 'ms'),
        virtual=True
    )

    debug("Concatenating...")
    da_multi = xd.concatenate(da_sequence, dim='time',
                              tolerance=np.timedelta64(tolerance, 'ms'), virtual=True)
    debug("Successfully read DataArray!")

    explore_da(da_multi, log=True)

    return da_multi


def read_file_list_optodas(files, tolerance=100):
    """
    Read all the files in a file list. Tolerance in [ms].
    """
    debug("Loading files sequentially...")
    das = []

    # Handle both single file and list
    if isinstance(files, str):
        files = [files]
    elif not isinstance(files, (list, tuple)):
        raise ValueError("files must be str or list/tuple of str")

    for i, f in enumerate(files, 1):
        debug(f"File {i}/{len(files)}: {os.path.basename(f)}")
        da = xd.open_dataarray(f, engine='asn')
        das.append(da)

    debug("Combining...")
    da_sequence = xd.combine_by_coords(
        das,
        dim='time',
        tolerance=np.timedelta64(tolerance, 'ms'),
        virtual=True
    )

    debug("Concatenating...")
    da_multi = xd.concatenate(da_sequence, dim='time',
                              tolerance=np.timedelta64(tolerance, 'ms'), virtual=True)
    debug("Successfully read DataArray!")

    explore_da(da_multi, log=True)

    return da_multi


def explore_da(da, log=True):
    # ========== TIMING INFO ==========
    t = da.coords["time"].values
    dt = t[1] - t[0]
    # Direct seconds as float
    dt_s = np.asarray(dt) / np.timedelta64(1, "s") if np.issubdtype(dt.dtype, np.timedelta64) else float(dt)
    t0_ms = t[0] if np.issubdtype(t.dtype, np.number) else np.datetime64(t[0], 'ms')
    t1_ms = t[-1] if np.issubdtype(t.dtype, np.number) else np.datetime64(t[-1], 'ms')

    # ========== SAMPLING INFO ==========
    x = da.coords["distance"].values
    dx = np.round(x[1] - x[0], 2)
    if log:
        debug(f"Sampling: {dt_s:.8f} s | Time: {t0_ms} to {t1_ms}")
        debug(f"Sampling: {dx} m | Distance: {x[0]:.2f} to {x[-1]:.2f} m")
        debug(f"Data shape: {da.shape}")
        debug(f"Data units: {da.name}")

    DAS_param = {'t': t, 'dt': dt_s, 'x': x, 'dx': dx, 'name': da.name}

    return DAS_param


def slice_sweep(da, df_row, t_sweep, xslice=None, taptest=None, delay=0):
    '''
    Select DAS data from the main file and slice it. Selecting taptest overwrites
    the first value of xslice.
    '''
    time1 = np.datetime64(df_row['datetime']) - np.timedelta64(int(delay*1000), 'ms')
    time2 = np.datetime64(time1 + np.timedelta64(t_sweep, 's'), 'ms')
    debug(f"Requested vibro time: {np.datetime64(df_row['datetime'])}")
    debug(f"Delayed time: {time1} to {time2}")

    if not df_row['process']:
        debug("---- WARNING! ---- \
              \nRequested time not marked True in the dataframe")

    if not df_row['Status']:
        debug("---- WARNING! ---- \
              \nRequested time not marked valid in the EVR file")

    if xslice is None:
        x1 = da.coords['distance'].values[0]
        x2 = da.coords['distance'].values[-1]
    else:
        x1 = xslice[0]
        x2 = xslice[1]

    if taptest is not None:
        x1 = taptest

    da_segment = da.sel(time=slice(time1, time2), distance=slice(x1, x2))
    da_segment.coords['distance'] = (da_segment.coords['distance'].values
                                     - da_segment.coords['distance'].values[0])
    da_segment.data = np.nan_to_num(da_segment.data, copy=True, nan=0.0,
                                    posinf=0.0, neginf=0.0)

    # debug(f"Extracted {t_sweep}s of sweep. File shape: {da_segment.shape}")
    return da_segment


def slice_hammer(da, df_row, t_sweep, xslice=None, taptest=None, delay=0):
    '''
    Select DAS data from the main file and slice it. Selecting taptest overwrites
    the first value of xslice.
    '''
    time1 = np.datetime64(df_row['datetime']) - np.timedelta64(int(delay*1000), 'ms')
    time2 = np.datetime64(time1 + np.timedelta64(t_sweep, 's'), 'ms')
    debug(f"Requested vibro time: {np.datetime64(df_row['datetime'])}")
    debug(f"Delayed time: {time1} to {time2}")

    if xslice is None:
        x1 = da.coords['distance'].values[0]
        x2 = da.coords['distance'].values[-1]
    else:
        x1 = xslice[0]
        x2 = xslice[1]

    if taptest is not None:
        x1 = taptest

    da_segment = da.sel(time=slice(time1, time2), distance=slice(x1, x2))
    da_segment.coords['distance'] = (da_segment.coords['distance'].values
                                     - da_segment.coords['distance'].values[0])
    da_segment.data = np.nan_to_num(da_segment.data, copy=True, nan=0.0,
                                    posinf=0.0, neginf=0.0)

    # debug(f"Extracted {t_sweep}s of sweep. File shape: {da_segment.shape}")
    return da_segment


def save_segment_perc(da, outdir, df_row, perc=None, colormap=None):
    '''
    Save data segment to file + PNG. MEMORY-SAFE.

    Parameters
    ----------
    da : xarray.DataArray
        Data to save and plot.
    outdir : str
        Output directory.
    df_row : pandas.Series or dict-like
        Must contain 'datetime' and 'FFID'.
    perc : float or int, optional
        If given in the range [0, 100], the colormap is clipped to the
        percentile range [perc, 100-perc]. Example:
        perc=2 -> vmin=p2, vmax=p98
    colormap : any matplotlib colormap
    '''
    # NetCDF
    timestamp = np.datetime64(df_row['datetime'], 's').astype(datetime).strftime('%Y%m%d_%H%M%S')
    filename = f"{df_row['FFID']}_{timestamp}.nc"
    filepath = os.path.join(outdir, filename)
    os.makedirs(outdir, exist_ok=True)
    da.to_netcdf(filepath, virtual=None)

    # PNG (isolated figure)
    png_dir = os.path.join(outdir, 'png')
    os.makedirs(png_dir, exist_ok=True)
    pngname = f"{df_row['FFID']}_{timestamp}.png"
    pngpath = os.path.join(png_dir, pngname)

    assert np.issubdtype(da.dtype, np.number), da.dtype

    plot_kwargs = dict(
        ax=None,
        yincrease=False,
        interpolation='bilinear',
        add_colorbar=False,
        cmap='RdBu'
    )

    if perc is not None:
        if not (0 <= perc <= 50):
            raise ValueError(f"perc must be between 0 and 50, got {perc}")

        data = np.asarray(da.values, dtype=float)
        data = data[np.isfinite(data)]

        if data.size == 0:
            raise ValueError("DataArray contains no finite values for percentile scaling")

        vmax = np.percentile(data, 100 - perc)
        vmin = -vmax

        plot_kwargs.update(vmin=vmin, vmax=vmax)
    else:
        plot_kwargs.update(robust=True)

    if colormap is not None:
        plot_kwargs.update(cmap=colormap)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    plot_kwargs["ax"] = ax

    da.plot(**plot_kwargs)

    # Save + IMMEDIATE CLEANUP
    fig.savefig(pngpath, dpi=150, bbox_inches='tight')

    # After savefig()
    plt.close(fig)
    del fig, ax
    gc.collect()

    print(f"Saved: {os.path.basename(filepath)} + {pngname}")
    return filepath, pngpath


def save_segment(da, outdir, df_row):
    '''
    Save data segment to file + PNG. MEMORY-SAFE.
    '''
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend (CRITICAL)

    # NetCDF
    timestamp = np.datetime64(df_row['datetime'], 's').astype(datetime).strftime('%Y%m%d_%H%M%S')
    filename = f"{df_row['FFID']}_{timestamp}.nc"
    filepath = os.path.join(outdir, filename)
    os.makedirs(outdir, exist_ok=True)
    da.to_netcdf(filepath, virtual=None)

    # PNG (isolated figure)
    png_dir = os.path.join(outdir, 'png')
    os.makedirs(png_dir, exist_ok=True)
    pngname = f"{df_row['FFID']}_{timestamp}.png"
    pngpath = os.path.join(png_dir, pngname)

    assert np.issubdtype(da.dtype, np.number), da.dtype

    # CREATE ISOLATED FIGURE
    fig = plt.figure(figsize=(8, 6))  # Explicit figure
    ax = fig.add_subplot(111)

    # Plot to specific ax (no global state)
    da.plot(ax=ax, robust=True, yincrease=False, interpolation='bilinear', add_colorbar=False)

    # Save + IMMEDIATE CLEANUP
    fig.savefig(pngpath, dpi=150, bbox_inches='tight')

    # After savefig()
    plt.close(fig)   #
    del fig, ax
    gc.collect()

    print(f"Saved: {os.path.basename(filepath)} + {pngname}")
    return filepath, pngpath


def interpolate_sweep(sweep_df, target_ms=2, sweep_type=2):
    '''
    interpolate sweep to 2 ms (default)
    '''
    sweep_df.iloc[:, 0].values
    time_sec = sweep_df.iloc[:, 0].values
    sweep_1ms = sweep_df.iloc[:, sweep_type].values
    time_ms = time_sec * 1000
    time_2ms = np.arange(time_ms[0], time_ms[-1], target_ms)
    sweep_2ms = np.interp(time_2ms, time_ms, sweep_1ms)
    return sweep_2ms, time_2ms


def corr_fft(da_segment, sweep):
    '''
    Cross-correlate using FFT method.
    '''
    dt = np.diff(da_segment.coords['time'].values).mean()
    dt = dt/np.timedelta64(1, 'ms')
    debug(f'dt = {dt}')
    lags = correlation_lags(da_segment.shape[0], sweep.shape[0], mode="full")*dt
    vr = sweep[::-1]
    return np.stack(
        [fftconvolve(da_segment.data[:, j], vr, mode="full")
         for j in range(da_segment.shape[1])],
        axis=1
    ), lags


def corr_direct(da_segment, sweep):
    dt = np.diff(da_segment.coords['time'].values).mean()
    dt = dt/np.timedelta64(1, 's')
    lags = correlation_lags(da_segment.shape[0], sweep.shape[0], mode="full")*dt
    return np.stack(
        [correlate(da_segment.data[:, j], sweep, mode="full")
         for j in range(da_segment.shape[1])],
        axis=1
    ), lags


def xcorr_to_dataarray(t_xc, xcorr_clip, spacing, fiber_length, name="xcorr", time_unit="s"):
    n_time, n_chan = xcorr_clip.shape  # 2D data first
    debug(xcorr_clip.shape)
    # Distance
    distance = spacing * np.arange(n_chan)
    distance = distance[distance <= fiber_length]
    if distance.size != n_chan:
        xcorr_clip = xcorr_clip[:, :distance.size]

    # xdas coords: dict of (dimname, coord_array) tuples
    coords = {
        "time": ("time", t_xc),
        "distance": ("distance", distance),
    }

    da_xc = xd.DataArray(
        data=xcorr_clip,  # 2D array
        dims=("time", "distance"),
        coords=coords,
        name=name,
        attrs={"time_unit": time_unit}
    )
    return da_xc


def normalize_rms(da):
    '''
    Normalize column-wise by robust RMS. Returns same type as input.
    '''
    clip_sigma = 3.0

    # Extract data (handles DataArray/numpy)
    if hasattr(da, 'dims'):
        data_2d = da.data
        is_da = True
    else:
        data_2d = da
        is_da = False

    # Per-channel clipping (±3σ)
    clip_lo = np.nanmean(data_2d, axis=0, keepdims=True) - \
        clip_sigma * np.nanstd(data_2d, axis=0, keepdims=True)
    clip_hi = np.nanmean(data_2d, axis=0, keepdims=True) + \
        clip_sigma * np.nanstd(data_2d, axis=0, keepdims=True)
    data_clipped = np.clip(data_2d, clip_lo, clip_hi)

    # RMS envelope (column-wise)
    rms_env = np.sqrt(np.mean(data_clipped ** 2, axis=0)
                      )  # shape (n_distance,)

    # Normalize (broadcasts correctly)
    # (n_time, 1) / (1, n_distance)
    data_norm = data_clipped / rms_env[np.newaxis, :]

    # Return same type
    if is_da:
        # Create new DataArray (don't modify original .data)
        da_norm = xd.DataArray(
            data_norm,
            dims=da.dims,
            coords=da.coords,
            name=da.name
        )

        return da_norm
    else:
        return data_norm


def read_stacks(paths):
    segments = []
    for i, p in enumerate(paths, 1):
        # print(f"  [{i}/{len(paths)}] {os.path.basename(p)}")
        segments.append(xd.open_dataarray(p))  # lazy load
    return segments


def stack_and_export(files_list, stack_outdir, rms=False):
    """
    Stack files + export NC + PNG.

    Parameters:
    -----------
    files_list : list
        List of .nc file paths
    stack_outdir : str
        Output folder
    method : str
        'median', 'mean', 'trimmed'
    normalize_rms : bool
        RMS normalize before stacking
    """
    print(f"Stacking {len(files_list)} files.")

    debug(f"Using RMS normalization before stacking? {rms}")
    # Load + optional normalize
    if rms:
        das = [normalize_rms(xd.open_dataarray(f)) for f in files_list]
    else:
        das = [xd.open_dataarray(f) for f in files_list]

    da_stack = xd.mean(xd.concatenate(das, dim='stack', virtual=False), dim='stack')  # full cube

    # Export
    os.makedirs(stack_outdir, exist_ok=True)

    # NC
    nc_path = os.path.join(stack_outdir, "final_stack.nc")
    da_stack.to_netcdf(nc_path)

    # PNG
    png_path = os.path.join(stack_outdir, "final_stack.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    da_stack.plot(ax=ax, robust=True, yincrease=False, cmap='RdBu',  interpolation='bilinear', add_colorbar=False)
    ax.set(title=f"Stack ({len(das)} FFIDs)",
           xlabel="Distance [m]", ylabel="Time [s]")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    debug(f" NC: {nc_path}")
    debug(f" PNG: {png_path}")
    return da_stack


def LS_stack_and_export(files_list, stack_outdir, LS=False):
    """
    Stack files + export NC + PNG.

    Parameters:
    -----------
    files_list : list
        List of .nc file paths
    stack_outdir : str
        Output folder
    method : str
        'median', 'mean', 'trimmed'
    normalize_rms : bool
        RMS normalize before stacking
    """
    print(f"Stacking {len(files_list)} files.")

    debug(f"Using LS-subtraction normalization before stacking? {LS}")
    # Load + optional normalize
    if LS:
        das = [LS_filtering(xd.open_dataarray(f), data2=None, dist_interval=[125, 175]) for f in files_list]
    else:
        das = [xd.open_dataarray(f) for f in files_list]

    da_stack = xd.mean(xd.concatenate(das, dim='stack', virtual=False), dim='stack')  # full cube

    if LS:
        da_stack = LS_filtering(da_stack, None, [100, 300])

    # Export
    os.makedirs(stack_outdir, exist_ok=True)

    # NC
    nc_path = os.path.join(stack_outdir, "final_stack.nc")
    da_stack.to_netcdf(nc_path)

    # PNG
    png_path = os.path.join(stack_outdir, "final_stack.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    da_stack.plot(ax=ax, robust=True, yincrease=False, cmap='RdBu',  interpolation='bilinear', add_colorbar=False)
    ax.set(title=f"Median Stack ({len(das)} FFIDs)",
           xlabel="Distance [m]", ylabel="Time [s]")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    debug(f" NC: {nc_path}")
    debug(f" PNG: {png_path}")
    return da_stack


def get_xdas_params(file):
    # This will get different for version 2.5 and higher!!
    with h5py.File(file, "r") as f:
        strain = f["fa1-21060040/Source1/Zone1/StrainRate"]
        headers = f["fa1-21060040/Source1/Zone1"]

        block_size = strain.shape[1]                      # block_time_size_wo
        block_overlap = int(headers.attrs["BlockOverlap"][0])  # percent
        version_bytes = headers.attrs["Version"]
        version_str = version_bytes.decode() if isinstance(version_bytes, (bytes, np.bytes_)) else str(version_bytes)

    block_time_size_wo = int(block_size)
    block_time_size_no = int(100 / (100 + block_overlap) * block_size)

    # Redundancy removal (your current code)
    to_remove = block_time_size_wo - block_time_size_no
    qotient, reminder = divmod(to_remove, 2)
    actual_data_st = max(0, qotient)
    actual_data_nd = block_time_size_wo - 1 - (reminder + qotient)

    # Decide timestamp position from Version
    try:
        major, minor, patch = (int(x) for x in version_str.split("."))
    except Exception:
        major = minor = patch = 0

    if (major, minor) >= (2, 3):
        # FEBUS ≥ 2.3: timestamp at center of block
        offset = block_time_size_wo // 2
    else:
        # older FEBUS: timestamp at start of block
        offset = 0

    # Overlaps to pass to XDAS
    overlaps = (actual_data_st, block_time_size_wo - 1 - actual_data_nd)
    debug(" Offset and overlaps read from the file:")
    debug(f" Overlaps: {overlaps}")
    debug(f" Offset: {offset}")
    return overlaps, offset


def get_xdas_params_murat(fname) -> tuple[tuple[int, int], int]:
    """
    Get the parameters for reading the FEBUS data.

    Parameters

    fname : str
        The name of the FEBUS file.

    Returns

    tuple[tuple[int, int], int]
        A tuple containing the overlaps and the offset.
    """
    headers = "fa1-21060040/Source1/Zone1"
    with h5py.File(fname, "r") as f:
        heads = f[headers]
        overlap = heads.attrs.get("BlockOverlap", [100])[0]
        extend = heads.attrs.get("Extent")
        block_rate_hz = heads.attrs.get("BlockRate")[0] / 1000
        block_time_size_wo = 1 + extend[3] - extend[2]

        t_spacing_s = (
            heads.attrs.get("Spacing")[1] / 1000
            if block_time_size_wo != 1
            else 1 / block_rate_hz
        )
        if block_time_size_wo == 1:
            block_time_size_no = 1
        else:
            if extend[2] == 0:
                block_time_size_no = int(
                    round(block_time_size_wo / (1 + (overlap / 100)), 0)
                )
            else:
                block_time_size_no = int(round(1 / (block_rate_hz * t_spacing_s), 0))

        to_remove = block_time_size_wo - block_time_size_no

        quotient, reminder = divmod(to_remove, 2)
        actual_data_st = int(max(0, quotient))
        actual_data_nd = block_time_size_wo - 1 - (reminder + quotient)
        overlaps = (actual_data_st, actual_data_st)
        offset = int((actual_data_nd - actual_data_st + 1) / 2)

        return (overlaps, overlaps[0] + offset)


def LS_filtering(data1, data2=None, dist_interval=None):
    """
    Least-squares subtraction of a reference signal from data.

    Parameters
    ----------
    data1         : xdas.DataArray, shape (nt, nx)
    data2         : xdas.DataArray, np.ndarray (1D or 2D), or None
                    Reference trace/array. If None, computed as mean of
                    data1 over dist_interval.
    dist_interval : tuple (d_min, d_max), same units as distance coord
                    Required only when data2 is None.

    Returns
    -------
    da_filtered   : xdas.DataArray with same coords as data1
    """

    # --- 1. Extract numpy from data1, remember coords for later ---
    is_xdas = hasattr(data1, 'coords')
    if is_xdas:
        arr1 = data1.values
        coords = data1.coords
        dims = data1.dims
        name = data1.name
        attrs = data1.attrs
    else:
        arr1 = np.asarray(data1)

    # --- 2. Build data2 if not provided ---
    if data2 is None:
        if dist_interval is None:
            raise ValueError("dist_interval=(d_min, d_max) must be provided when data2 is None")
        d_min, d_max = dist_interval
        ref_slice = data1.sel(distance=slice(d_min, d_max)).values  # (nt, n_ref)
        arr2 = ref_slice.mean(axis=1)                               # (nt,) mean trace
        debug(f"[LS] data2 = mean over distance [{d_min}, {d_max}] "
              f"({ref_slice.shape[1]} channels)")
    elif hasattr(data2, 'values'):
        arr2 = data2.values
    else:
        arr2 = np.asarray(data2)

    # --- 3. LS subtraction ---
    wtw = np.sum(arr2 * arr2)

    if arr2.ndim == 1:
        wtx = arr2 @ arr1                         # (nx,)
        amp = wtx / wtw                           # (nx,)
        LSdata = arr1 - arr2[:, None] * amp[None, :]  # (nt, nx)

    elif arr2.ndim == 2:
        if arr2.shape != arr1.shape:
            raise ValueError(f"Shape mismatch: data1 {arr1.shape} vs data2 {arr2.shape}")
        wtx = np.sum(arr2 * arr1, axis=0)         # (nx,)
        amp = wtx / wtw                           # (nx,)
        LSdata = arr1 - arr2 * amp                   # (nt, nx)

    else:
        raise ValueError(f"Unsupported data2.ndim={arr2.ndim}")

    # --- 4. Wrap back into xdas ---
    if is_xdas:
        return xd.DataArray(
            data=LSdata,
            coords=coords,
            dims=dims,
            name=name,
            attrs=attrs
        )
    return LSdata


def apply_lmo(da, v, reverse=False):
    """
    Apply Linear MoveOut (LMO) correction to an xdas DataArray.

    Parameters
    ----------
    da      : xdas.DataArray 
    v       : float (velocity in m/s)
    reverse : bool (if True, removes LMO)
    """
    # 1. Handle coordinates and dt
    offset = da.coords['distance'].values
    dt_raw = np.mean(np.diff(da.coords['time'].values))

    if dt_raw >= 1:
        dt = dt_raw / 1e3
        print(f"[LMO] dt = {dt_raw:.4g} ms detected -> using {dt:.6g} s")
    else:
        dt = dt_raw
        print(f"[LMO] dt = {dt:.6g} s detected -> using as-is")

    # 2. Compute shifts (integer indices)
    #    Round to nearest sample, cast to int64 for indexing
    shift = np.round(offset / v / dt).astype(np.int64)

    if reverse:
        shift = -shift

    # 3. Load data to RAM (necessary for advanced indexing)
    data = da.values
    nt, nx = data.shape

    # 4. Vectorized Roll using advanced indexing
    #    Create a column of row indices [0, 1, ..., nt-1]
    #    Broadcast add shifts to get source indices
    row_idx = (np.arange(nt)[:, None] + shift[None, :]) % nt

    #    Gather data: take from 'data' at 'row_idx' for each column
    datLMO = np.take_along_axis(data, row_idx, axis=0)
    # 5. Wrap back into xdas.DataArray
    #    We use the same coordinates and dims as the original 'da'
    da_lmo = xd.DataArray(
        data=datLMO,
        coords=da.coords,
        dims=da.dims,
        name=da.name,
        attrs=da.attrs
    )

    return da_lmo


def AGC(data, oper_len: float, basis: str = 'centred'):
    """
    Apply Automatic Gain Control to the data.
    Automatic Gain Control balances the gain based on the amplitude in a local
    window. The function is based on the AGC function from SeisSpace ProMAX.
    Scaling can be done based on the inverse of:
        mean
        median
        RMS
    The location of the window can be set as:
        trailing:
            Following the sample
        leading:
            Preceding the sample
        centred:
            The sample is located at the centre of the window
    See Also
    --------
    https://esd.halliburton.com/support/LSM/GGT/ProMAXSuite/ProMAX/5000/5000_8/Help/promax/agc.pdf
    Parameters
    ----------
    oper_len : float
        The operator length or also the window size of the local window.
    basis : str, optional
        The location of the window relative to the value that is changed.
        Can be trailing, leading or centred. The default is 'centred'.
    Returns
    -------
    omphalos.Gather
        The gather with AGC applied.
    """

    time = data.coords['time'].values
    dt = np.round(np.diff(time).mean(), 5)
    ns = data.shape[0]

    # The operator length in amount of data points
    oper_len_items = int(np.round(oper_len/dt+1))
    # Operator length must be uneven, so add 1 if it isn't
    if oper_len_items % 2 == 0:
        oper_len_items += 1
    # The convolution operator for the mean
    operator = np.ones(oper_len_items).T
    # Calculate how many data points are used for each point
    scal_vals = np.convolve(np.ones(ns), operator, 'full').T
    # Convolve the data with the operator and divide by the amount of points
    # used to get the mean
    convol = convolve2d(abs(data), operator[:, np.newaxis], 'full')/scal_vals[:, np.newaxis]
    convol = np.where(convol == 0, 1, convol)
    # Now snip out the relevant part for each method
    if basis == 'trailing':
        snipped = 1/convol[oper_len_items-1:, :]
    elif basis == 'leading':
        snipped = 1/convol[:-oper_len_items+1, :]
    elif basis == 'centred':
        snipped = 1/convol[int(oper_len_items/2):int(-oper_len_items/2), :]
    # Multiply the data with the scaling values
    new_data = data*snipped

    # if output_factor == True:
    # return new_data, snipped
    # else:
    # return new_data
    return new_data, snipped


def spher_div(data, factor=2):
    # Apply spherical divergence correction as t**2
    time = data.coords['time'].values
    data.data *= time[:, np.newaxis]**2  # spherical divergence
    return data


def nrms(da1, da2, dim='time'):
    diff = da1 - da2

    rms_diff = np.sqrt((diff ** 2).mean(dim=dim))
    rms_a = np.sqrt((da1 ** 2).mean(dim=dim))
    rms_b = np.sqrt((da2 ** 2).mean(dim=dim))

    nrms_val = 200 * rms_diff / (rms_a + rms_b)
    return nrms_val


def pred_calculate(da_stack, da_baseline, predlength):

    nlags = predlength // 2
    a = da_stack.data
    b = da_baseline.data                # FIX 4: was da_stack_baseline.data (global bug)

    # Remove mean
    a = a - np.mean(a, axis=0)
    b = b - np.mean(b, axis=0)

    N = a.shape[0]

    Rab = ss.fftconvolve(a, b[::-1], axes=0, mode="full")
    Raa = ss.fftconvolve(a, a[::-1], axes=0, mode="full")
    Rbb = ss.fftconvolve(b, b[::-1], axes=0, mode="full")

    mid = Rab.shape[0] // 2

    lag_window = np.arange(-nlags, nlags)
    indices = mid + lag_window

    Rab_win = Rab[indices]
    Raa_win = Raa[indices]
    Rbb_win = Rbb[indices]

    norm = (N - np.abs(lag_window))[:, None]

    Rab_win = Rab_win / norm
    Raa_win = Raa_win / norm
    Rbb_win = Rbb_win / norm

    numerator = np.sum(Rab_win**2, axis=0)
    denominator = np.sum(Raa_win * Rbb_win, axis=0)

    pred = 100 * numerator / (denominator + 1e-12)
    pred = np.minimum(pred, 100.0)

    print(f"PRED range: {pred.min():.1f}% to {pred.max():.1f}%")
    print(f"Any >1? {(numerator > denominator).sum()} channels")
    print(f"Max numerator/denom ratio: {(numerator/denominator).max():.4f}")

    return pred


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '--').count(os.sep)
        indent = '--' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = '--' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


def datetime_from_GPS(gps_time_us):
    """
    Convert GPS time in microseconds since GPS epoch
    (1980-01-06 00:00:00) to UTC datetime.

    Parameters
    ----------
    gps_time_us : int, float, pd.Series, array-like

    Returns
    -------
    pd.Timestamp or pd.Series
    """
    gps_epoch = pd.Timestamp("1980-01-06 00:00:00", tz="UTC")
    gps_utc_leap_seconds = 18

    return (
        gps_epoch
        + pd.to_timedelta(gps_time_us, unit="us")
        - pd.Timedelta(seconds=gps_utc_leap_seconds)
    )


# %%


def check_febus_params(file_path: str, zone: str = 'Zone1'):
    """
    Quick check of FEBUS file parameters for XDAS compatibility.

    Parameters
    ----------
    file_path : str
        Path to FEBUS HDF5 file
    zone : str
        Zone name (default 'Zone1')
    """
    from febus_optics_lib.reader import H5ReaderDas
    instance = H5ReaderDas(file_path)
    params = instance.param_dict[zone]

    print("="*60)
    print(f"FEBUS File Parameters: {zone}")
    print("="*60)

    # Basic info
    print(f"\nFile: {file_path}")
    print(f"Version: {params['version']}")
    print(f"Data type: {params['data_type']}")
    print(f"Timestamp position: {params.get('timestamp_position', 'unknown')}")

    # Timing
    print(f"\nTiming:")
    print(f"  Sampling rate: {params['sampling_rate']:.1f} Hz")
    print(f"  Temporal spacing: {params['temporal_spacing']*1000:.4f} ms")
    print(f"  Block rate: {params['block_rate']:.1f} Hz")
    print(f"  Nyquist: {params['nyquist']:.1f} Hz")

    # Block structure
    print(f"\nBlock structure:")
    print(f"  Block time size (with overlap): {params['block_time_size_wo']} samples")
    print(f"  Block time size (no overlap): {params['block_time_size_no']} samples")
    print(f"  Samples to remove: {params['block_time_size_wo'] - params['block_time_size_no']}")
    print(f"  Number of blocks: {params['timestamp_vect_size']}")

    # Calculate overlaps
    to_remove = params['block_time_size_wo'] - params['block_time_size_no']
    left, remainder = divmod(to_remove, 2)
    right = left + remainder

    print(f"\nXDAS Parameters:")
    print(f"  overlaps=(left, right): ({left}, {right})")

    # Calculate offset based on version
    version_parts = params['version'].split('.')
    try:
        version_float = float(f"{version_parts[0]}.{version_parts[1]}")
    except:
        version_float = 2.3

    timestamp_pos = params.get('timestamp_position', 'middle')

    if params['block_time_size_wo'] == 1:
        offset = 0
    elif timestamp_pos == 'start':
        offset = 0
    elif timestamp_pos == 'middle':
        if version_float < 2.5:
            offset = (params['block_time_size_no'] - 1) // 2
        else:
            offset = (params['block_time_size_wo'] // 2) - left
    else:
        offset = 0

    print(f"  offset: {offset}")

    # Spatial
    print(f"\nSpatial:")
    print(f"  Distance start: {params['distance_start']:.1f} m")
    print(f"  Distance end: {params['distance_end']:.1f} m")
    print(f"  Distance spacing: {params['distance_spacing']:.2f} m")
    print(f"  Number of channels: {params['distance_vect_size']}")
    print(f"  Gauge length: {params['gauge_length']:.1f} m")

    # Data extent
    print(f"\nData extent:")
    print(f"  Time range: {params['utc_date_start']} {params['utc_time_start']}")
    print(f"           to {params['utc_date_end']} {params['utc_time_end']}")
    print(f"  Duration: {params['timestamp_end'] - params['timestamp_start']:.1f} s")

    # Memory estimate
    n_samples_concat = params['timestamp_vect_size'] * params['block_time_size_no']
    n_channels = params['distance_vect_size']
    memory_mb = (float(n_samples_concat) * float(n_channels) * 4) / (1024**2)  # float32

    print(f"\nMemory estimate (full file, concat):")
    print(f"  Shape: ({n_samples_concat}, {n_channels})")
    print(f"  Size: {memory_mb:.1f} MB")

    print("\n" + "="*60)

    # Return parameters for programmatic use
    return {
        'overlaps': (left, right),
        'offset': offset,
        'version': params['version'],
        'timestamp_position': timestamp_pos,
        'sampling_rate': params['sampling_rate'],
        'block_time_size_wo': params['block_time_size_wo'],
        'block_time_size_no': params['block_time_size_no'],
    }


def pred_calculate(da_stack, da_baseline, predlength):

    nlags = predlength // 2
    a = da_stack.data
    b = da_baseline.data

    # Remove mean
    a = a - np.mean(a, axis=0)
    b = b - np.mean(b, axis=0)

    N = a.shape[0]

    Rab = ss.fftconvolve(a, b[::-1], axes=0, mode="full")
    Raa = ss.fftconvolve(a, a[::-1], axes=0, mode="full")
    Rbb = ss.fftconvolve(b, b[::-1], axes=0, mode="full")

    mid = Rab.shape[0] // 2
    # lags = np.arange(-mid, mid+1)

    # Only keep desired window
    lag_window = np.arange(-nlags, nlags)
    indices = mid + lag_window

    Rab_win = Rab[indices]
    Raa_win = Raa[indices]
    Rbb_win = Rbb[indices]

    # Proper lag-dependent normalization
    norm = (N - np.abs(lag_window))[:, None]

    Rab_win = Rab_win / norm
    Raa_win = Raa_win / norm
    Rbb_win = Rbb_win / norm

    numerator = np.sum(Rab_win**2, axis=0)
    denominator = np.sum(Raa_win * Rbb_win, axis=0)

    pred = 100 * numerator / (denominator + 1e-12)
    pred = np.minimum(pred, 100.0)

    print(f"PRED range: {pred.min():.1f}% to {pred.max():.1f}%")

    print(f"Any >1? {(numerator > denominator).sum()} channels")
    print(f"Max numerator/denom ratio: {(numerator/denominator).max():.4f}")

    return pred


def nrms(da1, da2, dim="time"):
    diff = da1 - da2

    rms_diff = np.sqrt((diff**2).mean(dim=dim))
    rms_a = np.sqrt((da1**2).mean(dim=dim))
    rms_b = np.sqrt((da2**2).mean(dim=dim))

    nrms_val = 200 * rms_diff / (rms_a + rms_b)
    return nrms_val


def read_optodas_conversion_params(path):
    """
    Read OptoDAS metadata needed for conversion to strain rate.
    """
    with h5py.File(path, "r") as h5:
        params = {
            "wavelength": float(h5["header/wavelength"][()]),
            "refractive_index": float(np.ravel(h5["cableSpec/refractiveIndexes"][()])[0]),
            "zeta": float(h5["cableSpec/zeta"][()]),
            "gauge_length": float(h5["header/gaugeLength"][()]),
            "dt": float(h5["header/dt"][()]),
        }
    return params


def optodas_to_strain(da, params, axis=0, detrend_mean=True):
    """
    Convert OptoDAS dphi data to strain rate.
    """
    dphi = np.asarray(da.data, dtype=np.float64)

    if detrend_mean:
        dphi = dphi - np.mean(dphi, axis=axis, keepdims=True)

    factor = params["wavelength"] / (
        4.0 * np.pi * params["refractive_index"] * params["zeta"] * params["gauge_length"]
    )
    strain_rate = factor * dphi

    coords = {
        "time": ("time", da.coords["time"].values),
        "distance": ("distance", da.coords["distance"].values),
    }

    return xd.DataArray(
        data=strain_rate,
        dims=("time", "distance"),
        coords=coords,
        name="strain_rate",
        attrs={"time_unit": "s"},
    )


def wiggle(da, ax=None, sf=1.0, color="k", left_color="w", right_color="k",
           cmap=None, cmap_dir="h", cmap_side="right",
           alpha=1.0, time_dim="time", distance_dim="distance"):
    """Wiggle plot from an xdas.DataArray. ax = wiggle(da).

    sf=1 -> each trace's max deviation (peak-to-peak) equals one channel
    spacing. Each trace normalized to its own max absolute amplitude.

    cmap : str | None
        If set, lobes are filled with a colormap instead of flat colors
        (MATLAB 'X'/'*' modes).
    cmap_dir : {'h', 'v'}
        'h' (horizontal-layered, MATLAB 'X'): color varies by amplitude/
        depth -- i.e. layers run horizontally across the section.
        'v' (vertical-layered, MATLAB '*'): color varies by lateral
        position within the lobe (offset -> tip), independent of amplitude
        -- i.e. layers run vertically, like a gradient fill of the lobe.
    cmap_side : {'right', 'left', 'both'}
        Which lobe(s) get the colormap fill (MATLAB single vs doubled
        'X'/'XX' or '*'/'**' controls left vs right vs both).
    """
    if hasattr(da, "dims"):
        if da.dims[0] != time_dim:
            da = da.transpose(time_dim, distance_dim)
        data = np.asarray(da.values)
        tt = np.asarray(da.coords[time_dim].values)
        xx = np.asarray(da.coords[distance_dim].values)
    else:
        data, tt, xx = da, np.arange(da.shape[0]), np.arange(da.shape[1])

    ts = float(np.min(np.diff(xx))) if len(xx) > 1 else 1.0

    trace_max = np.max(np.abs(data), axis=0)
    trace_max[trace_max == 0] = 1.0
    data_s = data / trace_max * (ts / 2.0) * sf

    ax = ax or plt.gca()

    if cmap is not None:
        cmap_obj = plt.get_cmap(cmap)
        amp_norm = plt.Normalize(vmin=-np.max(np.abs(data)), vmax=np.max(np.abs(data)))
        pos_norm = plt.Normalize(vmin=0.0, vmax=1.0)

    for i in range(data.shape[1]):
        trace, offset = data_s[:, i], xx[i]
        zc = np.where(np.diff(np.signbit(trace)))[0]
        if len(zc):
            x1, x2 = tt[zc], tt[zc + 1]
            y1, y2 = trace[zc], trace[zc + 1]
            tz = x1 - y1 / ((y2 - y1) / (x2 - x1))
            tt_s, tr_s = np.split(tt, zc + 1), np.split(trace, zc + 1)
            tt_zi, trace_zi = tt_s[0], tr_s[0]
            for k in range(len(tz)):
                tt_zi = np.hstack((tt_zi, tz[k:k+1], tt_s[k+1]))
                trace_zi = np.hstack((trace_zi, [0], tr_s[k+1]))
        else:
            tt_zi, trace_zi = tt, trace

        def fill_side(where_mask, use_cmap):
            if not use_cmap:
                col = right_color if where_mask == "right" else left_color
                if col in (None, "none"):
                    return
                mask = trace_zi >= 0 if where_mask == "right" else trace_zi <= 0
                ax.fill_betweenx(tt_zi, offset, trace_zi + offset, where=mask,
                                 facecolor=col, alpha=alpha, linewidth=0)
                return

            mask = trace_zi >= 0 if where_mask == "right" else trace_zi <= 0
            idx = np.where(mask)[0]
            if len(idx) < 2:
                return

            # horizontal-layered: color = amplitude at each depth sample
            polys, colors = [], []
            for j in range(len(trace_zi) - 1):
                if not (mask[j] and mask[j + 1]):
                    continue
                polys.append([(offset, tt_zi[j]), (trace_zi[j] + offset, tt_zi[j]),
                              (trace_zi[j + 1] + offset, tt_zi[j + 1]), (offset, tt_zi[j + 1])])
                val = 0.5 * (trace_zi[j] + trace_zi[j + 1]) / ((ts / 2.0) * sf) * trace_max[i]
                colors.append(cmap_obj(amp_norm(val)))
            ax.add_collection(PolyCollection(polys, facecolors=colors,
                                             edgecolors="none", alpha=alpha))

        use_cmap_right = cmap is not None and cmap_side in ("right", "both")
        use_cmap_left = cmap is not None and cmap_side in ("left", "both")
        fill_side("right", use_cmap_right)
        fill_side("left", use_cmap_left)

        if color not in (None, "none"):
            ax.plot(trace_zi + offset, tt_zi, color=color, linewidth=0.5, alpha=alpha)

    ax.set_xlim(xx[0] - ts, xx[-1] + ts)
    ax.set_ylim(tt[0], tt[-1])
    ax.invert_yaxis()
    return ax


def cmap_and_wiggles(da2d, sf=2.5, alpha=0.5, ax=None,  **kwargs):
    da2d = normalize_rms(da2d)
    da2d.plot(ax=ax, **kwargs)
    dx = np.min(np.diff(da2d.coords["distance"].values))
    da_shifted = da2d.assign_coords({"distance": da2d.coords["distance"].values-dx/2})
    axs = wiggle(da_shifted, sf=sf, right_color='k', left_color=None, alpha=alpha, ax=ax)   # MATLAB 'XX'
    dist = da_shifted.coords['distance'].values
    axs.set_xlim(dist[0], dist[-1]+dx)
    return axs


def phase_shift(da, dist=(10, 30), f=(0, 100), vel=(0, 200),
                n_vel=100, sigma=None,  n_bins=None, off_neg=False, agc=None, time_dim="time", distance_dim="distance"):
    """Compute the MASW phase-shift dispersion image from an xdas.DataArray.

    Parameters
    ----------
    da : xdas.DataArray
        2-D array with time and distance dims/coords.
    dist : tuple(float, float)
        Distance window (min, max) selecting the receiver spread to use.
    f : tuple(float, float)
        Frequency search window in Hz, applied after the FFT.
    vel : tuple(float, float)
        Phase velocity search window in m/s.
    n_vel : int
        Number of velocity samples in the scanning grid (mesh resolution).
    n_bins : int | None
        Number of FFT bins (passed to xdas.fft.rfft's `n`), controlling
        frequency resolution. If None, uses the native trace length.
    time_dim, distance_dim : str
        Coordinate/dimension names in `da`.
    sigma : int | None
        std smoothing of a spatial gaussian filter. If none then no smoothing
    off_neg : bool
        Mark true if moveout is to the left. if True then search is in the other direction
    agc : float | None
        if true, performs 
    Returns
    -------
    xdas.DataArray
        Dispersion image with dims ('velocity', 'frequency'), sliced to the
        requested `f` window and normalized per-frequency-column (each
        frequency's max amplitude across velocities = 1).
    """
    # 1. Select the distance window directly via xdas
    if agc is not None:
        da, rev = AGC(da, agc)
    da_win = da.sel({distance_dim: slice(dist[0], dist[1])})
    x = np.asarray(da_win.coords[distance_dim].values)
    if x.size < 2:
        raise ValueError("dist window selects fewer than 2 channels")

    # 2. FFT along time -> frequency, using xdas's own rfft
    fft_kwargs = {"dim": {time_dim: "frequency"}}
    if n_bins is not None:
        fft_kwargs["n"] = n_bins
    DA = xfft.rfft(da_win, **fft_kwargs)

    # 3. Restrict to the requested frequency window
    DA = DA.sel({"frequency": slice(f[0], f[1])})
    freqs = np.asarray(DA.coords["frequency"].values)
    Ufft = np.asarray(DA.values)  # shape (n_f, n_x) if freq is dim0, else transpose below
    if DA.dims[0] != "frequency":
        DA = DA.transpose("frequency", distance_dim)
        Ufft = np.asarray(DA.values)

    # 4. Normalize spectrum (phase-only)
    Rnorm = Ufft / (np.abs(Ufft) + 1e-30)

    # 5. Velocity scanning grid
    v0 = vel[0] if vel[0] > 0 else 1e-6
    ct = np.linspace(v0, vel[1], n_vel)
    w = 2 * np.pi * freqs

    AsSum = np.zeros((len(freqs), n_vel))
    for ii in range(len(w)):
        if off_neg is True:
            phase = np.exp(-1j * w[ii] * (x[:, None] / ct[None, :]))  # (n_x, n_vel)
        else:
            phase = np.exp(1j * w[ii] * (x[:, None] / ct[None, :]))  # (n_x, n_vel)
        As = phase * Rnorm[ii, :, None]
        AsSum[ii, :] = np.abs(np.sum(As, axis=0))

    if sigma is not None:
        AsSum = gaussian_filter(AsSum, sigma=sigma)

    # 6. Wrap result back into an xdas.DataArray
    result = xd.DataArray(
        data=AsSum.T,  # (n_vel, n_f)
        coords={
            "velocity": ("velocity", ct),
            "frequency": ("frequency", freqs),
        },
        dims=("velocity", "frequency"),
    )
    return result
