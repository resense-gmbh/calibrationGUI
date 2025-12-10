
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# ---------------- CONFIG: set this to the folder that contains HEX8, HEX10, ... ----------------
# From your earlier run log, paths looked like:
# C:\Users\DaryapS\WIKA\EM-CEO-ORG-Resense - Files\110_Manufacturing\Calibration\00_KPI\Si\HEX8\Master_KPI_Hex8.xlsx
base_dir = r"C:\Users\DaryapS\WIKA\EM-CEO-ORG-Resense - Files\110_Manufacturing\Calibration\00_KPI\Si"


# Column mapping (current layout): B = sensor id/name, D = quarter, E+ = KPIs
SENSOR_COL_IDX = 1   # Column B
QUARTER_COL_IDX = 3  # Column D
KPI_START_IDX   = 4  # Column E onward

# Plot style: 'radar' or 'bars'
PLOT_STYLE = 'radar'

# Radar readability controls
MAX_SENSORS_ON_RADAR = 8      # overlay at most N sensors per quarter
NORMALIZE_RADAR      = True   # normalize per KPI to [0,1] within each quarter

# ===============================================================

AXES = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

# --------------------------- helpers ---------------------------
def list_sensor_folders(root):
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Base directory not found: {root}")
    items = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    return sorted([d for d in items if d.upper().startswith("HEX")])

def choose_by_number(prompt, items):
    choice = input(prompt).strip()
    if not choice.isdigit():
        raise ValueError("Please enter a number.")
    idx = int(choice) - 1
    if idx < 0 or idx >= len(items):
        raise ValueError("Invalid numeric choice.")
    return items[idx]

def find_master_excel(sensor_dir):
    """Silently pick the first Master_KPI_*.xlsx in folder, else first .xlsx."""
    xlsx_files = [f for f in os.listdir(sensor_dir) if f.lower().endswith(".xlsx")]
    if not xlsx_files:
        raise FileNotFoundError(f"No Excel files found in: {sensor_dir}")
    preferred = [f for f in xlsx_files if f.lower().startswith("master_kpi_")]
    selected = preferred[0] if preferred else xlsx_files[0]
    path = os.path.join(sensor_dir, selected)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Selected Excel file not found: {path}")
    return path  # no menu, no extra messages

def read_kpi_excel(path):
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(f"Excel file not found: {path}")
    # If your workbook has title/group rows, try header=1 or header=2
    df = pd.read_excel(path, engine="openpyxl")  # or header=1 / header=2
    df.columns = [str(c).strip() for c in df.columns]
    return df

def build_quarter_sensor_matrix(df, quarter_value, kpi_cols):
    """Return sensors list, non-empty KPI subset, and KPI matrix for the selected quarter."""
    qmask = df.iloc[:, QUARTER_COL_IDX].astype(str).str.strip().str.lower() == str(quarter_value).strip().lower()
    dfq = df[qmask].copy()
    if dfq.empty:
        raise ValueError(f"No rows found for Quarter='{quarter_value}'")
    sensors = dfq.iloc[:, SENSOR_COL_IDX].astype(str).str.strip().tolist()
    kpi_numeric = dfq[kpi_cols].apply(pd.to_numeric, errors="coerce")
    non_empty_kpis = kpi_numeric.columns[kpi_numeric.notna().any(axis=0)].tolist()
    kpi_numeric = kpi_numeric[non_empty_kpis]
    return sensors, non_empty_kpis, kpi_numeric

def align_kpis(kpis1, kpis2):
    return [k for k in kpis1 if k in kpis2]

def build_kpi_groups(kpi_cols):
    """Group KPI columns by axis suffix, and collect base parameter names per axis."""
    groups = {ax: [] for ax in AXES}
    params_by_axis = {ax: set() for ax in AXES}
    for col in kpi_cols:
        col_str = str(col).strip()
        for ax in AXES:
            suffix = "_" + ax
            if col_str.endswith(suffix):
                groups[ax].append(col_str)
                params_by_axis[ax].add(col_str[:-len(suffix)])  # base param
                break
    all_params = sorted({p for ax in AXES for p in params_by_axis[ax]})
    return groups, params_by_axis, all_params

def print_kpi_groups(groups):
    print("\nKPI groups by axis:")
    for i, ax in enumerate(AXES, start=1):
        members = groups[ax]
        print(f"{i}. {ax}  ({len(members)} KPIs)")
        if members:
            preview = ", ".join(members[:6])
            if len(members) > 6:
                preview += ", ..."
            print(f"   e.g.: {preview}")

def _tokenize(raw):
    return [t.strip() for t in raw.split(",") if t.strip()]

def choose_axes(groups):
    print_kpi_groups(groups)
    raw = input("\nEnter axis to include (numbers or names, comma-separated; e.g., '1,2' or 'Fx,Fy'): ").strip()
    tokens = _tokenize(raw)
    chosen = set()
    lower_to_axis = {ax.lower(): ax for ax in AXES}
    for t in tokens:
        if t.isdigit():
            idx = int(t) - 1
            if 0 <= idx < len(AXES):
                chosen.add(AXES[idx])
            else:
                raise ValueError(f"Invalid axis number: {t}")
        else:
            if t.lower() in lower_to_axis:
                chosen.add(lower_to_axis[t.lower()])
            else:
                raise ValueError(f"Unknown axis: {t}")
    if not chosen:
        raise ValueError("No axis selected.")
    return sorted(chosen, key=lambda a: AXES.index(a))

def choose_params_for_axes(params_by_axis, selected_axes):
    """Ask for params by numbers/names, OR '*' to include all (no 'Use ALL?' prompt)."""
    # Use UNION of params available in selected axes
    param_union = sorted({p for ax in selected_axes for p in params_by_axis[ax]})
    print("\nAvailable parameter names (across selected axes):")
    for i, p in enumerate(param_union, start=1):
        print(f"{i}. {p}")
    raw = input("\nEnter parameter(s) (numbers or names, comma-separated) or '*' for ALL: ").strip()
    if raw in ("*", "",):
        return param_union  # all, silently
    tokens = _tokenize(raw)
    selected_params = []
    lower_map = {p.lower(): p for p in param_union}
    for t in tokens:
        if t.isdigit():
            idx = int(t) - 1
            if 0 <= idx < len(param_union):
                selected_params.append(param_union[idx])
            else:
                raise ValueError(f"Invalid parameter number: {t}")
        else:
            if t.lower() in lower_map:
                selected_params.append(lower_map[t.lower()])
            else:
                raise ValueError(f"Parameter not found: {t}")
    # dedupe
    seen, result = set(), []
    for p in selected_params:
        if p not in seen:
            result.append(p); seen.add(p)
    if not result:
        raise ValueError("No parameter selected.")
    return result

def expand_params_to_columns(selected_axes, selected_params, groups):
    """Turn (axes, params) into concrete KPI column names that exist in the sheet."""
    selected_cols, seen = [], set()
    for ax in selected_axes:
        existing = set(groups[ax])
        for p in selected_params:
            c = f"{p}_{ax}"
            if c in existing and c not in seen:
                selected_cols.append(c)
                seen.add(c)
    if not selected_cols:
        raise ValueError("No KPI columns matched the chosen axes/parameters.")
    return selected_cols

# --------------------------- plotting ---------------------------
def normalize_matrix_01(mat):
    """Per-KPI min-max normalization to [0,1] within a quarter, for radar readability."""
    m = mat.copy()
    for c in m.columns:
        col = m[c]
        mn, mx = col.min(skipna=True), col.max(skipna=True)
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            # constant or all NaN -> set to zeros
            m[c] = 0.0
        else:
            m[c] = (col - mn) / (mx - mn)
    return m

def radar_subplot(ax, sensors, mat, title):
    """Draw a radar chart for a quarter; limit to top-N sensors by KPI average."""
    if mat.shape[1] < 2:
        # radar isn't useful with 1 KPI -> fallback to bars
        _bar_single_quarter(ax, sensors, mat, title)
        return

    # limit number of sensors for readability
    avg_scores = mat.mean(axis=1, skipna=True).fillna(0.0).values
    order = np.argsort(avg_scores)[::-1]  # descending
    pick = order[:MAX_SENSORS_ON_RADAR]
    mat = mat.iloc[pick].reset_index(drop=True)
    sensors = [sensors[i] for i in pick]

    # normalize (optional)
    if NORMALIZE_RADAR:
        mat = normalize_matrix_01(mat)

    labels = list(mat.columns)
    n = len(labels)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    cmap = mpl.colormaps.get_cmap('tab20')
    for i, sensor in enumerate(sensors):
        vals = mat.iloc[i].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=cmap(i % cmap.N), linewidth=2, label=str(sensor))
        ax.fill(angles, vals, color=cmap(i % cmap.N), alpha=0.20)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=8)

def _bar_single_quarter(ax, sensors, mat, title):
    """Fallback bar chart when only one KPI is selected."""
    x = np.arange(len(sensors))
    width = 0.8
    cmap = mpl.colormaps.get_cmap('tab20')
    vals = mat.iloc[:, 0].values if mat.shape[1] else np.zeros(len(sensors))
    ax.bar(x, vals, width=width, color=[cmap(i % cmap.N) for i in range(len(sensors))])
    ax.set_xticks(x)
    ax.set_xticklabels(sensors, rotation=45, ha='right', fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(mat.columns[0] if mat.shape[1] else "Value")
    ax.grid(axis='y', linestyle=':', alpha=0.4)

def plot_quarter_bars(ax, sensors, mat, title):
    """Grouped bars: sensors per KPI."""
    num_kpis = len(mat.columns)
    x = np.arange(num_kpis)
    width = 0.8 / max(1, len(sensors))
    cmap = mpl.colormaps.get_cmap('tab20')
    for i, sensor in enumerate(sensors):
        vals = mat.iloc[i].values if i < len(mat) else np.zeros(num_kpis)
        ax.bar(x + i*width, vals, width=width, label=str(sensor), color=cmap(i % cmap.N))
    ax.set_xticks(x + (width * (len(sensors)-1) / 2))
    ax.set_xticklabels(list(mat.columns), rotation=45, ha='right')
    ax.set_title(title)
    ax.set_xlabel("KPI")
    ax.set_ylabel("Value")
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.15), ncol=2, fontsize=8)

# ============================= main =============================
if __name__ == "__main__":
    # 1) Sensor folder (number-only)
    sensor_folders = list_sensor_folders(base_dir)
    if not sensor_folders:
        raise RuntimeError("No HEX* sensor folders found under base directory.")
    print("\nAvailable sensor families (folders):")
    for i, d in enumerate(sensor_folders, start=1):
        print(f"{i}. {d}")
    sensor_folder_choice = choose_by_number("\nEnter number of sensor family folder: ", sensor_folders)
    sensor_folder_path = os.path.join(base_dir, sensor_folder_choice)

    # 2) Excel auto-pick (no menu/prompt)
    excel_path = find_master_excel(sensor_folder_path)
    print(f"\nUsing Excel file: {excel_path}")

    # 3) Read & basic validation
    df = read_kpi_excel(excel_path)
    if df.shape[1] < KPI_START_IDX + 1:
        raise ValueError("The Excel file doesn't have enough columns for B/D/E+. Please check layout.")
    quarters = sorted(df.iloc[:, QUARTER_COL_IDX].dropna().astype(str).str.strip().unique().tolist())
    kpi_all = df.columns[KPI_START_IDX:]

    # 4) Choose quarters
    print("\nAvailable quarters (from Column D):")
    for i, q in enumerate(quarters, start=1):
        print(f"{i}. {q}")
    quarter1 = choose_by_number("\nEnter number for FIRST quarter: ", quarters)

    print("\nAvailable quarters (from Column D):")
    for i, q in enumerate(quarters, start=1):
        print(f"{i}. {q}")
    quarter2 = choose_by_number("Enter number for SECOND quarter: ", quarters)

    # 5) Axis-aware KPI selection (no 'Use ALL?' prompt — use '*' for all)
    groups, params_by_axis, _all_params = build_kpi_groups(kpi_all)
    selected_axes   = choose_axes(groups)
    selected_params = choose_params_for_axes(params_by_axis, selected_axes)  # '*' returns all
    kpi_selected    = expand_params_to_columns(selected_axes, selected_params, groups)

    print("\nSelected KPI columns:")
    for c in kpi_selected:
        print(" -", c)

    # 6) Build matrices
    sensors_q1, kpis_q1, mat_q1 = build_quarter_sensor_matrix(df, quarter1, kpi_selected)
    sensors_q2, kpis_q2, mat_q2 = build_quarter_sensor_matrix(df, quarter2, kpi_selected)
    kpis_common = align_kpis(kpis_q1, kpis_q2)
    if not kpis_common:
        raise ValueError("The selected KPI(s) have no numeric data in common between the two quarters.")
    mat_q1 = mat_q1[kpis_common].reset_index(drop=True)
    mat_q2 = mat_q2[kpis_common].reset_index(drop=True)

    # 7) Plot (radar default; bars fallback if only one KPI or if you set PLOT_STYLE='bars')
    if PLOT_STYLE.lower() == 'radar':
        fig, axes = plt.subplots(1, 2, figsize=(max(12, len(kpis_common)*1.0), 7), subplot_kw=dict(polar=True))
        ax1, ax2 = axes
        radar_subplot(ax1, sensors_q1, mat_q1, f"Quarter: {quarter1} ({sensor_folder_choice})")
        radar_subplot(ax2, sensors_q2, mat_q2, f"Quarter: {quarter2} ({sensor_folder_choice})")
        fig.suptitle(f"Radar — {', '.join(kpis_common)}\nAll sensors in {quarter1} vs {quarter2}", fontsize=13)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(max(12, len(kpis_common)*1.0), 6), sharey=True)
        ax1, ax2 = axes
        plot_quarter_bars(ax1, sensors_q1, mat_q1, f"Quarter: {quarter1} ({sensor_folder_choice})")
        plot_quarter_bars(ax2, sensors_q2, mat_q2, f"Quarter: {quarter2} ({sensor_folder_choice})")
        fig.suptitle(f"Bars — {', '.join(kpis_common)}\nAll sensors in {quarter1} vs {quarter2}", fontsize=13)

    fig.tight_layout()
    plt.show()  # show only; do not save
