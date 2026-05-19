import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, io
from datetime import datetime

from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, PageBreak
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER

PAGE = landscape(A4)
W, H = PAGE

# ─────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────
PASS_COLOR = colors.HexColor('#1a7a4a')
FAIL_COLOR = colors.HexColor('#c0392b')
LIGHT_GRAY = colors.HexColor('#f4f6f7')
MID_GRAY   = colors.HexColor('#bdc3c7')
DARK       = colors.HexColor('#2c3e50')
NAVY       = colors.HexColor('#0d2b55')


# ─────────────────────────────────────────────
# 1. CSV IMPORT
# ─────────────────────────────────────────────

def load_data(filepath):
    """Load a CSV file and return a DataFrame.
    Edit TARGET_COL and PREDICTOR_COLS in __main__ to match your data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[DATA] Loaded {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"[DATA] Columns: {list(df.columns)}\n")
    return df


# ─────────────────────────────────────────────
# 2. MODEL SELECTION
#    Single predictor: Linear, Poly deg 2-4, Logarithmic, Exponential,
#                      Log-Log, Reciprocal, Spline, Lag-1
#    Multi predictor:  Linear, Interaction (all pairwise x_i * x_j)
#    All ranked by original-scale BIC for direct comparability.
# ─────────────────────────────────────────────

def _model_type_label(degree):
    """Return a human-readable label for polynomial model types."""
    return 'Linear' if degree == 1 else f'Polynomial (deg {degree})'


def _equation_str(model, target_col, predictor_cols):
    """Build a human-readable equation string from fitted coefficients."""
    params = model.params
    parts  = []
    if 'const' in params:
        parts.append(f"{params['const']:.3f}")
    for name, coef in params.items():
        if name == 'const':
            continue
        sign = '+' if coef >= 0 else '-'
        parts.append(f"{sign} {abs(coef):.4f}*{name}")
    return f"{target_col} = " + " ".join(parts)


def select_top_models(df, target_col, predictor_cols, max_degree=4):
    from patsy import dmatrix

    results = []
    single  = (len(predictor_cols) == 1)
    col     = predictor_cols[0] if single else None

    def _add(model_type, mdl, Xc, orig_fitted=None):
        """Append a result dict; orig_fitted supplies back-transformed predictions
        for models where the response was transformed (Exponential, Log-Log)."""
        res  = mdl.resid
        sw_p = shapiro(res)[1]
        bp_p = het_breuschpagan(res, mdl.model.exog)[1]
        results.append({
            'Degree':       1,
            'ModelType':    model_type,
            'Equation':     _equation_str(mdl, target_col, list(Xc.columns[1:])),
            'R2':           round(mdl.rsquared, 4),
            'Adj_R2':       round(mdl.rsquared_adj, 4),
            'AIC':          round(mdl.aic, 2),
            'BIC':          round(mdl.bic, 2),
            'SW_p':         round(sw_p, 4),
            'BP_p':         round(bp_p, 4),
            '_model':       mdl,
            '_X':           Xc,
            '_orig_fitted': orig_fitted,
        })

    # ── Polynomials deg 1-4 ──────────────────────────────────────────────────
    for deg in range(1, max_degree + 1):
        X = df[predictor_cols].copy()
        if deg > 1 and single:
            for d in range(2, deg + 1):
                X[f'{col}^{d}'] = df[col] ** d
        Xc  = sm.add_constant(X)
        mdl = sm.OLS(df[target_col], Xc).fit()
        res = mdl.resid
        results.append({
            'Degree':       deg,
            'ModelType':    _model_type_label(deg),
            'Equation':     _equation_str(mdl, target_col, predictor_cols),
            'R2':           round(mdl.rsquared, 4),
            'Adj_R2':       round(mdl.rsquared_adj, 4),
            'AIC':          round(mdl.aic, 2),
            'BIC':          round(mdl.bic, 2),
            'SW_p':         round(shapiro(res)[1], 4),
            'BP_p':         round(het_breuschpagan(res, mdl.model.exog)[1], 4),
            '_model':       mdl,
            '_X':           Xc,
            '_orig_fitted': None,
        })

    if single:
        x_vals = df[col].values
        y_vals = df[target_col].values
        x_pos  = (x_vals > 0).all()
        y_pos  = (y_vals > 0).all()

        # ── Logarithmic  y ~ ln(x) ───────────────────────────────────────────
        if x_pos:
            try:
                Xc = sm.add_constant(pd.DataFrame({f'ln({col})': np.log(x_vals)}))
                _add('Logarithmic', sm.OLS(y_vals, Xc).fit(), Xc)
            except Exception:
                pass

        # ── Exponential  ln(y) ~ x ───────────────────────────────────────────
        if y_pos:
            try:
                Xc  = sm.add_constant(df[[col]])
                mdl = sm.OLS(np.log(y_vals), Xc).fit()
                _add('Exponential', mdl, Xc,
                     orig_fitted=np.exp(mdl.fittedvalues.values))
            except Exception:
                pass

        # ── Log-Log  ln(y) ~ ln(x)  (power law) ─────────────────────────────
        if x_pos and y_pos:
            try:
                Xc  = sm.add_constant(pd.DataFrame({f'ln({col})': np.log(x_vals)}))
                mdl = sm.OLS(np.log(y_vals), Xc).fit()
                _add('Log-Log (Power Law)', mdl, Xc,
                     orig_fitted=np.exp(mdl.fittedvalues.values))
            except Exception:
                pass

        # ── Reciprocal  y ~ 1/x ──────────────────────────────────────────────
        if (x_vals != 0).all():
            try:
                Xc = sm.add_constant(pd.DataFrame({f'1/{col}': 1.0 / x_vals}))
                _add('Reciprocal', sm.OLS(y_vals, Xc).fit(), Xc)
            except Exception:
                pass

        # ── Natural cubic spline (3 interior knots at quartiles) ─────────────
        try:
            knots     = np.quantile(x_vals, [0.25, 0.50, 0.75])
            spline_df = dmatrix(f'cr(x, knots={list(knots)})',
                                {'x': x_vals}, return_type='dataframe').iloc[:, 1:]
            Xc = sm.add_constant(spline_df)
            _add('Spline (Natural Cubic)', sm.OLS(y_vals, Xc).fit(), Xc)
        except Exception:
            pass

        # ── Lag-1  y_t ~ x_t + x_{t-1} ──────────────────────────────────────
        try:
            lag_col = f'{col}_lag1'
            X_lag   = pd.DataFrame({col: x_vals[1:], lag_col: x_vals[:-1]})
            Xc      = sm.add_constant(X_lag)
            _add('Lag-1', sm.OLS(y_vals[1:], Xc).fit(), Xc)
        except Exception:
            pass

    else:
        # ── Multi-predictor: pairwise interaction terms ───────────────────────
        try:
            from itertools import combinations
            X_int = df[predictor_cols].copy()
            for c1, c2 in combinations(predictor_cols, 2):
                X_int[f'{c1}*{c2}'] = df[c1] * df[c2]
            Xc  = sm.add_constant(X_int)
            mdl = sm.OLS(df[target_col], Xc).fit()
            res = mdl.resid
            results.append({
                'Degree':       1,
                'ModelType':    'Interaction',
                'Equation':     _equation_str(mdl, target_col, list(X_int.columns)),
                'R2':           round(mdl.rsquared, 4),
                'Adj_R2':       round(mdl.rsquared_adj, 4),
                'AIC':          round(mdl.aic, 2),
                'BIC':          round(mdl.bic, 2),
                'SW_p':         round(shapiro(res)[1], 4),
                'BP_p':         round(het_breuschpagan(res, mdl.model.exog)[1], 4),
                '_model':       mdl,
                '_X':           Xc,
                '_orig_fitted': None,
            })
        except Exception:
            pass

    # ── Rank by original-scale BIC ───────────────────────────────────────────
    # BIC = n*ln(RSS/n) + k*ln(n) computed on raw y regardless of response
    # transformation, making all model families directly comparable.
    actual = df[target_col].values

    def _orig_scale_bic(row):
        of  = row['_orig_fitted']
        mdl = row['_model']
        if of is not None:
            fitted, y = np.asarray(of), actual
        else:
            fitted = mdl.fittedvalues.values
            y      = actual[1:] if row['ModelType'] == 'Lag-1' else actual
        rss = np.sum((y - fitted) ** 2)
        k   = int(mdl.df_model) + 1
        n   = len(fitted)
        return n * np.log(rss / n) + k * np.log(n) if rss > 0 else -np.inf

    res_df = pd.DataFrame(results)
    res_df['_bic_orig'] = res_df.apply(_orig_scale_bic, axis=1)
    combined = res_df.sort_values('_bic_orig').reset_index(drop=True)

    best_row = combined.iloc[0]
    return (combined.head(10).copy(),
            best_row['_model'], best_row['_X'],
            int(best_row['Degree']), str(best_row['ModelType']))


# ─────────────────────────────────────────────
# 3. DIAGNOSTICS
# ─────────────────────────────────────────────

def run_diagnostics(model, X):
    residuals = model.resid
    n         = len(residuals)

    vif_rows = []
    for i, col in enumerate(X.columns):
        if col == 'const':
            continue
        v = variance_inflation_factor(X.values, i)
        vif_rows.append((col, round(v, 2), 'PASS' if v < 5 else 'FAIL'))

    sw_stat, sw_p                    = shapiro(residuals)
    jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(residuals)
    bp_stat, bp_p, _, _              = het_breuschpagan(residuals, model.model.exog)
    dw                               = durbin_watson(residuals)

    cooks_d       = model.get_influence().cooks_distance[0]
    threshold     = 4 / n
    n_influential = int(np.sum(cooks_d > threshold))

    return dict(
        n=n,
        r2=model.rsquared, adj_r2=model.rsquared_adj,
        f_stat=model.fvalue, f_pval=model.f_pvalue,
        df_model=int(model.df_model), df_resid=int(model.df_resid),
        f_pass=(model.f_pvalue < 0.05),
        vif_rows=vif_rows,
        vif_pass=all(r[2] == 'PASS' for r in vif_rows),
        sw_stat=sw_stat, sw_p=sw_p, sw_pass=(sw_p > 0.05),
        jb_stat=jb_stat, jb_p=jb_p, jb_skew=jb_skew, jb_kurt=jb_kurt,
        jb_pass=(jb_p > 0.05),
        bp_stat=bp_stat, bp_p=bp_p, bp_pass=(bp_p > 0.05),
        dw=dw, dw_pass=(1.5 < dw < 2.5),
        cooks_d=cooks_d, cooks_threshold=threshold,
        n_influential=n_influential,
        residuals=residuals,
        fitted=model.fittedvalues,
    )


# ─────────────────────────────────────────────
# 4. HELPER — figure → bytes
# ─────────────────────────────────────────────

def _fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


# ─────────────────────────────────────────────
# 5. OLS PLOT (single predictor: scatter + fit curve;
#              multi predictor: actual vs fitted scatter)
# ─────────────────────────────────────────────

def make_ols_plot(df, model, target_col, predictor_cols, degree, diag,
                  model_type='Linear'):
    from patsy import dmatrix

    fig, ax = plt.subplots(figsize=(10, 4.5))
    actual  = df[target_col].values

    if len(predictor_cols) == 1:
        x_col    = predictor_cols[0]
        x_vals   = df[x_col].values
        x_smooth = np.linspace(x_vals.min(), x_vals.max(), 400)
        params   = model.params

        ax.scatter(x_vals, actual, color='#4a90d9', alpha=0.5,
                   s=28, label='Actual', zorder=3)

        try:
            a_val = params.get('const', 0)
            b_key = [k for k in params.index if k != 'const'][0]

            if model_type == 'Exponential':
                y_smooth = np.exp(a_val + params[b_key] * x_smooth)

            elif model_type == 'Logarithmic':
                y_smooth = a_val + params[b_key] * np.log(np.maximum(x_smooth, 1e-9))

            elif model_type == 'Log-Log (Power Law)':
                y_smooth = np.exp(a_val + params[b_key] * np.log(np.maximum(x_smooth, 1e-9)))

            elif model_type == 'Reciprocal':
                safe     = np.where(np.abs(x_smooth) < 1e-9, np.nan, x_smooth)
                y_smooth = a_val + params[b_key] * (1.0 / safe)

            elif model_type == 'Spline (Natural Cubic)':
                knots         = np.quantile(x_vals, [0.25, 0.50, 0.75])
                spline_smooth = dmatrix(f'cr(x, knots={list(knots)})',
                                        {'x': x_smooth},
                                        return_type='dataframe').iloc[:, 1:]
                Xs = sm.add_constant(spline_smooth)
                Xs = Xs.reindex(columns=model.model.exog_names, fill_value=0)
                y_smooth = model.predict(Xs)

            elif model_type == 'Lag-1':
                sort_idx = np.argsort(x_vals[1:])
                ax.plot(x_vals[1:][sort_idx], model.fittedvalues.values[sort_idx],
                        color='#e74c3c', linewidth=2,
                        label=f'OLS Fit ({model_type})', zorder=4)
                ax.set_xlabel(x_col, fontsize=11, fontweight='bold')
                ax.set_ylabel(target_col, fontsize=11, fontweight='bold')
                ax.set_title(f'OLS Regression: {x_col} → {target_col}  [{model_type}]',
                             fontsize=12, fontweight='bold')
                ax.legend(fontsize=9, prop={'weight': 'bold'})
                ax.text(0.02, 0.97,
                        f'R²={diag["r2"]:.4f}   Adj R²={diag["adj_r2"]:.4f}',
                        transform=ax.transAxes, fontsize=9, va='top', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
                fig.tight_layout()
                return _fig_to_bytes(fig)

            else:
                # Linear / Polynomial
                y_smooth = np.full_like(x_smooth, params.get('const', 0))
                for d in range(1, degree + 1):
                    term = f'{x_col}^{d}' if d > 1 else x_col
                    if term in params:
                        y_smooth = y_smooth + params[term] * (x_smooth ** d)

            ax.plot(x_smooth, y_smooth, color='#e74c3c', linewidth=2,
                    label=f'OLS Fit ({model_type})', zorder=4)

        except Exception:
            sort_idx = np.argsort(x_vals)
            ax.plot(x_vals[sort_idx], diag['fitted'].values[sort_idx],
                    color='#e74c3c', linewidth=2,
                    label=f'OLS Fit ({model_type}) [fallback]', zorder=4)

        ax.set_xlabel(x_col, fontsize=11, fontweight='bold')
        ax.set_ylabel(target_col, fontsize=11, fontweight='bold')
        ax.set_title(f'OLS Regression: {x_col} → {target_col}  [{model_type}]',
                     fontsize=12, fontweight='bold')

    else:
        # Multi-predictor: actual vs fitted scatter
        fitted_y = diag['fitted'].values
        mn, mx   = min(fitted_y.min(), actual.min()), max(fitted_y.max(), actual.max())
        ax.scatter(fitted_y, actual, color='#4a90d9', alpha=0.5, s=28, zorder=3)
        ax.plot([mn, mx], [mn, mx], color='#e74c3c', linewidth=2,
                linestyle='--', label='Perfect Fit')
        ax.set_xlabel('Fitted Values', fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual Values', fontsize=11, fontweight='bold')
        ax.set_title('Actual vs Fitted', fontsize=12, fontweight='bold')

    ax.legend(fontsize=9, prop={'weight': 'bold'})
    ax.text(0.02, 0.97,
            f'R²={diag["r2"]:.4f}   Adj R²={diag["adj_r2"]:.4f}',
            transform=ax.transAxes, fontsize=9, va='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    fig.tight_layout()
    return _fig_to_bytes(fig)


# ─────────────────────────────────────────────
# 6. DIAGNOSTIC PLOTS
# ─────────────────────────────────────────────

def make_diagnostic_plots(diag):
    residuals = diag['residuals']
    fitted    = diag['fitted']
    cooks_d   = diag['cooks_d']
    threshold = diag['cooks_threshold']
    n         = diag['n']

    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    # Residuals vs Fitted
    ax1  = fig.add_subplot(gs[0, 0])
    infl = cooks_d > threshold
    ax1.scatter(fitted, residuals,
                s=np.where(infl, 50, 25), c='#4a90d9', alpha=0.65, zorder=3)
    for idx in np.where(infl)[0]:
        ax1.scatter(fitted.iloc[idx], residuals.iloc[idx],
                    s=260, facecolors='none', edgecolors='#e74c3c',
                    linewidths=1.8, zorder=4)
    ax1.axhline(0, color='black', lw=1, ls='--')
    ax1.set_xlabel('Fitted Values', fontsize=9, fontweight='bold')
    ax1.set_ylabel('Residuals', fontsize=9, fontweight='bold')
    ax1.set_title("Residuals vs Fitted\n(red circles = influential, Cook's D > 4/n)",
                  fontsize=9, fontweight='bold')

    # Q-Q
    ax2 = fig.add_subplot(gs[0, 1])
    sm.qqplot(residuals, line='s', ax=ax2, alpha=0.55,
              marker='o', markerfacecolor='#4a90d9', markeredgewidth=0)
    ax2.set_title("Q-Q Plot (Normality of Residuals)", fontsize=9, fontweight='bold')
    ax2.set_xlabel('Theoretical Quantiles', fontsize=9, fontweight='bold')
    ax2.set_ylabel('Sample Quantiles', fontsize=9, fontweight='bold')

    # Cook's Distance
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(range(n), cooks_d,
            color=['#e74c3c' if c > threshold else '#4a90d9' for c in cooks_d],
            alpha=0.75, width=0.8)
    ax3.axhline(threshold, color='#e74c3c', ls='--', lw=1.2,
                label=f'Threshold 4/n={threshold:.4f}')
    ax3.set_xlabel('Observation Index', fontsize=9, fontweight='bold')
    ax3.set_ylabel("Cook's Distance", fontsize=9, fontweight='bold')
    ax3.set_title("Cook's Distance (Influential Observations)", fontsize=9, fontweight='bold')
    ax3.legend(fontsize=8, prop={'weight': 'bold'})

    # Residual histogram
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(residuals, bins=20, color='#4a90d9', edgecolor='white', alpha=0.8)
    ax4.set_xlabel('Residual Value', fontsize=9, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=9, fontweight='bold')
    ax4.set_title('Residual Distribution', fontsize=9, fontweight='bold')

    fig.suptitle('OLS Diagnostic Plots', fontsize=13, fontweight='bold')
    return _fig_to_bytes(fig)


# ─────────────────────────────────────────────
# 7. CORRELATION PLOT
#    1 predictor  → None  (OLS plot is sufficient)
#    2 predictors → 3-D scatter + OLS regression plane
#    3+ predictors→ pairwise correlation heatmap
# ─────────────────────────────────────────────

def make_correlation_plot(df, target_col, predictor_cols, model=None):
    n_pred = len(predictor_cols)

    if n_pred == 1:
        return None

    if n_pred == 2:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        x1_col, x2_col = predictor_cols
        x1 = df[x1_col].values
        x2 = df[x2_col].values
        y  = df[target_col].values

        fig = plt.figure(figsize=(9, 6))
        ax  = fig.add_subplot(111, projection='3d')

        ax.scatter(x1, x2, y, color='#2ecc71', s=30, alpha=0.75,
                   depthshade=True, label='Samples', zorder=5)

        x1_r = np.linspace(x1.min(), x1.max(), 30)
        x2_r = np.linspace(x2.min(), x2.max(), 30)
        X1g, X2g = np.meshgrid(x1_r, x2_r)

        if model is not None:
            grid_df = pd.DataFrame({x1_col: X1g.ravel(), x2_col: X2g.ravel()})
            grid_X  = sm.add_constant(grid_df, has_constant='add')
            for col in model.model.exog_names:
                if col not in grid_X.columns:
                    grid_X[col] = 0.0
            Zg = model.predict(grid_X[model.model.exog_names]).values.reshape(X1g.shape)
        else:
            Xc = sm.add_constant(pd.DataFrame({x1_col: x1, x2_col: x2}))
            fb = sm.OLS(y, Xc).fit()
            p  = fb.params
            Zg = p.get('const', 0) + p.get(x1_col, 0) * X1g + p.get(x2_col, 0) * X2g

        ax.plot_surface(X1g, X2g, Zg, color='#aec6f0', alpha=0.45,
                        linewidth=0, antialiased=True)
        ax.set_xlabel(x1_col, fontsize=9, fontweight='bold', labelpad=8)
        ax.set_ylabel(x2_col, fontsize=9, fontweight='bold', labelpad=8)
        ax.set_zlabel(target_col, fontsize=9, fontweight='bold', labelpad=8)
        ax.set_title(f'3-D Regression: {x1_col} & {x2_col}  →  {target_col}',
                     fontsize=11, fontweight='bold', pad=12)
        ax.legend(fontsize=9, loc='upper left')
        fig.tight_layout()
        return _fig_to_bytes(fig)

    # 3+ predictors: correlation heatmap
    from matplotlib.colors import LinearSegmentedColormap

    cols_all = predictor_cols + [target_col]
    corr     = df[cols_all].corr()
    n        = len(cols_all)

    FIG_W   = 13.0
    FIG_H   = max(3.2, min(4.6, n * 0.42))
    cell_fs = max(6.5, 9.5 - n * 0.25)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor('white')

    _teal_orange = LinearSegmentedColormap.from_list(
        'teal_orange',
        ['#8B4513', '#CC7722', '#F5DEB3', '#80CBC4', '#00695C'],
        N=256
    )

    im = ax.imshow(corr.values, cmap=_teal_orange, vmin=-1, vmax=1, aspect='auto')

    cbar_ticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02, ticks=cbar_ticks)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.ax.set_yticklabels(
        [('-1' if t == -1 else ('1' if t == 1 else f'{t:.2f}')) for t in cbar_ticks],
        fontsize=7)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cols_all, rotation=45, ha='right',
                       fontsize=cell_fs, fontweight='bold')
    ax.set_yticklabels(cols_all, fontsize=cell_fs, fontweight='bold')
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            txt_color = 'white' if abs(val) > 0.60 else '#2c3e50'
            label = '-1' if val == -1 else ('1' if val == 1 else f'{val:.2f}')
            ax.text(j, i, label, ha='center', va='center',
                    fontsize=cell_fs, fontweight='bold', color=txt_color)

    ax.set_title('')
    fig.tight_layout(pad=0.8)
    return _fig_to_bytes(fig)


# ─────────────────────────────────────────────
# 8. PDF STYLES & HELPERS
# ─────────────────────────────────────────────

def _styles():
    return {
        'title': ParagraphStyle('title', fontSize=17, leading=21,
                                textColor=NAVY, alignment=TA_CENTER,
                                fontName='Helvetica-Bold', spaceAfter=4),
        'sub':   ParagraphStyle('sub', fontSize=9, leading=13,
                                textColor=colors.HexColor('#7f8c8d'),
                                alignment=TA_CENTER, spaceAfter=10),
        'h2':    ParagraphStyle('h2', fontSize=11, leading=15,
                                textColor=NAVY, fontName='Helvetica-Bold',
                                spaceBefore=8, spaceAfter=4),
        'h3':    ParagraphStyle('h3', fontSize=9.5, leading=13,
                                textColor=NAVY, fontName='Helvetica-Bold',
                                spaceBefore=6, spaceAfter=3),
        'body':  ParagraphStyle('body', fontSize=8.5, leading=12, textColor=DARK),
        'note':  ParagraphStyle('note', fontSize=7.5, leading=11,
                                textColor=colors.HexColor('#555555'), leftIndent=6),
        'small': ParagraphStyle('small', fontSize=7, leading=10,
                                textColor=colors.HexColor('#7f8c8d')),
        'interp_head': ParagraphStyle('interp_head', fontSize=10, leading=14,
                                      textColor=NAVY, fontName='Helvetica-Bold',
                                      spaceBefore=5, spaceAfter=2),
        'interp_body': ParagraphStyle('interp_body', fontSize=8.5, leading=13,
                                      textColor=DARK, leftIndent=8, spaceAfter=3),
        'hdr_cell': ParagraphStyle('hdr_cell', fontSize=8, leading=11,
                                   textColor=colors.white, fontName='Helvetica-Bold'),
    }


def _status(passed):
    txt, c = ('PASS', PASS_COLOR) if passed else ('FAIL', FAIL_COLOR)
    return Paragraph(
        f'<font color="#{c.hexval()[2:]}"><b>{txt}</b></font>',
        ParagraphStyle('s', fontSize=8, leading=11, alignment=TA_CENTER))


def _tbl_style(header_color=None):
    hc = header_color or NAVY
    return TableStyle([
        ('FONTNAME',       (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE',       (0, 0), (-1, -1), 8),
        ('BACKGROUND',     (0, 0), (-1,  0), hc),
        ('TEXTCOLOR',      (0, 0), (-1,  0), colors.white),
        ('FONTNAME',       (0, 0), (-1,  0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ('GRID',           (0, 0), (-1, -1), 0.4, MID_GRAY),
        ('VALIGN',         (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING',    (0, 0), (-1, -1), 5),
        ('RIGHTPADDING',   (0, 0), (-1, -1), 5),
        ('TOPPADDING',     (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING',  (0, 0), (-1, -1), 3),
    ])


# ─────────────────────────────────────────────
# 9. INTERPRETATION HELPERS
# ─────────────────────────────────────────────

def _interpretation_bullets(diag, best_degree, target_col, predictor_cols, best_model_type=''):
    """Return a list of (heading, text) tuples for the interpretation section."""
    bullets = []

    is_poly     = any(best_model_type.startswith(p) for p in ('Linear', 'Polynomial'))
    model_label = f'degree {best_degree}' if is_poly else best_model_type or f'degree {best_degree}'
    r2, ar2     = diag['r2'], diag['adj_r2']
    fit_quality = 'strong' if ar2 >= 0.80 else ('moderate' if ar2 >= 0.50 else 'weak')

    bullets.append(("Overall Fit",
        f"The best model ({model_label}) explains {r2*100:.1f}% of variance in "
        f"{target_col} (R\u00b2 = {r2:.4f}, Adj R\u00b2 = {ar2:.4f}), indicating a {fit_quality} fit."))

    bullets.append(("Model Significance (F-Test)",
        f"F({diag['df_model']}, {diag['df_resid']}) = {diag['f_stat']:.3f}, "
        f"p = {diag['f_pval']:.4f}. The overall regression is "
        f"{'statistically significant (p < 0.05)' if diag['f_pass'] else 'not significant (p \u2265 0.05)'}."))

    bullets.append(("Normality of Residuals (Shapiro-Wilk)",
        f"W = {diag['sw_stat']:.4f}, p = {diag['sw_p']:.4f}. "
        + ("Residuals appear normally distributed — the normality assumption is met."
           if diag['sw_pass'] else
           "Residuals deviate from normality (p \u2264 0.05). Consider robust SE or transformation.")))

    bullets.append(("Homoscedasticity (Breusch-Pagan)",
        f"BP stat = {diag['bp_stat']:.4f}, p = {diag['bp_p']:.4f}. "
        + ("Residual variance appears constant (homoscedastic)."
           if diag['bp_pass'] else
           "Evidence of heteroscedasticity detected. Consider WLS or robust standard errors.")))

    dw_val = diag['dw']
    bullets.append(("Autocorrelation (Durbin-Watson)",
        f"DW = {dw_val:.4f}. "
        + ("No significant autocorrelation detected."
           if diag['dw_pass'] else
           f"Possible {'positive' if dw_val < 1.5 else 'negative'} autocorrelation detected. "
           "Consider time-series methods.")))

    if len(diag['vif_rows']) > 1:
        vif_max = max(r[1] for r in diag['vif_rows'])
        bullets.append(("Multicollinearity (VIF)",
            f"Max VIF = {vif_max:.2f}. "
            + ("No multicollinearity concern (all VIF < 5)."
               if diag['vif_pass'] else
               "High multicollinearity detected — consider dropping correlated predictors "
               "or using ridge regression.")))

    ni = diag['n_influential']
    bullets.append(("Influential Observations (Cook's Distance)",
        "No influential observations detected." if ni == 0 else
        f"{ni} influential observation(s) detected (Cook's D > {diag['cooks_threshold']:.4f}). "
        "Review flagged points for data entry errors or genuine outliers."))

    return bullets


# ─────────────────────────────────────────────
# 10. BUILD PDF
# ─────────────────────────────────────────────

def build_pdf(diag, top10, ols_png, diag_png, out_path,
              target_col, predictor_cols, best_degree, best_model_type='Linear',
              corr_png=None):
    """
    Page layout by predictor count:
      1 predictor  → OLS plot + scorecard | interpretation + model table | diagnostic plots
      2 predictors → OLS plot + 3-D scatter side-by-side + scorecard | interp + table | diag plots
      3+ predictors→ correlation heatmap alone | scorecard | interp + model table | diagnostic plots
    """
    n_pred = len(predictor_cols)

    doc   = SimpleDocTemplate(out_path, pagesize=PAGE,
                              leftMargin=1.4*cm, rightMargin=1.4*cm,
                              topMargin=1.1*cm,  bottomMargin=1.1*cm)
    S     = _styles()
    story = []
    uw    = W - 2.8*cm

    def _page_header():
        story.append(Paragraph("OLS Regression - Diagnostic Audit Report", S['title']))
        story.append(Paragraph(
            f"Target: <b>{target_col}</b>  |  "
            f"Predictors: <b>{', '.join(predictor_cols)}</b>  |  "
            f"Best Model: <b>{best_model_type}</b>  |  n = <b>{diag['n']}</b>",
            S['sub']))
        story.append(HRFlowable(width=uw, thickness=1.5, color=NAVY, spaceAfter=8))

    # ═══════════════════════════════════════════
    # PAGE 1 — Visualisation
    # ═══════════════════════════════════════════
    _page_header()

    if n_pred == 1 or corr_png is None:
        story.append(Paragraph(
            f"OLS Regression Plot - Best Model ({best_model_type})", S['h2']))
        story.append(RLImage(ols_png, width=uw, height=uw * 0.34))

    elif n_pred == 2:
        story.append(Paragraph(
            f"OLS Regression Plot ({best_model_type})  &  3-D Scatter + Regression Plane",
            S['h2']))
        half     = uw / 2 - 4
        side_tbl = Table(
            [[RLImage(ols_png,  width=half, height=half * 0.68),
              RLImage(corr_png, width=half, height=half * 0.68)]],
            colWidths=[half, half])
        side_tbl.setStyle(TableStyle([
            ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING',   (0, 0), (-1, -1), 2),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 2),
            ('TOPPADDING',    (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(side_tbl)

    else:
        story.append(Paragraph(
            "Correlation Heatmap — All Predictors & Target", S['h2']))
        story.append(RLImage(corr_png, width=uw, height=uw * 0.44))
        story.append(PageBreak())
        _page_header()

    story.append(Spacer(1, 6))

    # ═══════════════════════════════════════════
    # DIAGNOSTIC SCORECARD
    # ═══════════════════════════════════════════
    story.append(Paragraph(
        f"Diagnostic Scorecard - Best Model ({best_model_type})", S['h2']))

    vif_max  = max((r[1] for r in diag['vif_rows']), default=0)
    vif_pass = diag['vif_pass']

    score_rows = [[
        Paragraph('<b>Test</b>',      S['hdr_cell']),
        Paragraph('<b>Statistic</b>', S['hdr_cell']),
        Paragraph('<b>p-value</b>',   S['hdr_cell']),
        Paragraph('<b>Threshold</b>', S['hdr_cell']),
        Paragraph('<b>Result</b>',    S['hdr_cell']),
        Paragraph('<b>Notes</b>',     S['hdr_cell']),
    ]]

    score_rows += [
        [Paragraph("F-Test (Overall)", S['note']),
         Paragraph(f"{diag['f_stat']:.4f}", S['small']),
         Paragraph(f"{diag['f_pval']:.4f}", S['small']),
         Paragraph("p < 0.05", S['small']),
         _status(diag['f_pass']),
         Paragraph(f"df model={diag['df_model']}, resid={diag['df_resid']}", S['small'])],

        [Paragraph("Normality - Shapiro-Wilk", S['note']),
         Paragraph(f"{diag['sw_stat']:.4f}", S['small']),
         Paragraph(f"{diag['sw_p']:.4f}",   S['small']),
         Paragraph("p > 0.05", S['small']),
         _status(diag['sw_pass']),
         Paragraph("Tests residual normality", S['small'])],

        [Paragraph("Normality - Jarque-Bera", S['note']),
         Paragraph(f"{diag['jb_stat']:.4f}", S['small']),
         Paragraph(f"{diag['jb_p']:.4f}",   S['small']),
         Paragraph("p > 0.05", S['small']),
         _status(diag['jb_pass']),
         Paragraph(f"Skew={diag['jb_skew']:.3f}, Kurt={diag['jb_kurt']:.3f}", S['small'])],

        [Paragraph("Homoscedasticity - Breusch-Pagan", S['note']),
         Paragraph(f"{diag['bp_stat']:.4f}", S['small']),
         Paragraph(f"{diag['bp_p']:.4f}",   S['small']),
         Paragraph("p > 0.05", S['small']),
         _status(diag['bp_pass']),
         Paragraph("Tests constant variance", S['small'])],

        [Paragraph("Autocorrelation - Durbin-Watson", S['note']),
         Paragraph(f"{diag['dw']:.4f}", S['small']),
         Paragraph("-", S['small']),
         Paragraph("1.5 – 2.5", S['small']),
         _status(diag['dw_pass']),
         Paragraph("Tests residual independence", S['small'])],
    ]

    if diag['vif_rows']:
        score_rows.append([
            Paragraph("Multicollinearity - Max VIF", S['note']),
            Paragraph(f"{vif_max:.2f}", S['small']),
            Paragraph("-", S['small']),
            Paragraph("< 5", S['small']),
            _status(vif_pass),
            Paragraph(f"Predictors: {', '.join(r[0] for r in diag['vif_rows'])}", S['small']),
        ])

    score_rows.append([
        Paragraph("Influential Obs - Cook's D", S['note']),
        Paragraph(f"{diag['n_influential']} flagged", S['small']),
        Paragraph("-", S['small']),
        Paragraph(f"4/n = {diag['cooks_threshold']:.4f}", S['small']),
        _status(diag['n_influential'] == 0),
        Paragraph("Red circles on residual plot", S['small']),
    ])

    sc_tbl = Table(score_rows,
                   colWidths=[uw*p for p in [0.25, 0.10, 0.09, 0.10, 0.08, 0.38]],
                   repeatRows=1)
    sc_tbl.setStyle(_tbl_style(NAVY))
    story.append(sc_tbl)
    story.append(Spacer(1, 6))
    story.append(PageBreak())

    # ═══════════════════════════════════════════
    # PAGE — Interpretation + Model Comparison
    # ═══════════════════════════════════════════
    story.append(Paragraph("OLS Regression - Interpretation & Model Comparison", S['title']))
    story.append(Paragraph(
        f"Target: <b>{target_col}</b>  |  "
        f"Predictors: <b>{', '.join(predictor_cols)}</b>  |  "
        f"Best Model: <b>{best_model_type}</b>  |  n = <b>{diag['n']}</b>",
        S['sub']))
    story.append(HRFlowable(width=uw, thickness=1.5, color=NAVY, spaceAfter=8))

    story.append(Paragraph("Interpretation of Results", S['h2']))
    interp_rows = [
        [Paragraph(f"<b>{h}</b>", S['interp_head']), Paragraph(t, S['interp_body'])]
        for h, t in _interpretation_bullets(diag, best_degree, target_col,
                                            predictor_cols, best_model_type)
    ]
    interp_tbl = Table(interp_rows, colWidths=[uw * 0.25, uw * 0.75])
    interp_tbl.setStyle(TableStyle([
        ('FONTNAME',       (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE',       (0, 0), (-1, -1), 8.5),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, LIGHT_GRAY]),
        ('GRID',           (0, 0), (-1, -1), 0.3, MID_GRAY),
        ('VALIGN',         (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',    (0, 0), (-1, -1), 6),
        ('RIGHTPADDING',   (0, 0), (-1, -1), 6),
        ('TOPPADDING',     (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',  (0, 0), (-1, -1), 4),
        ('BACKGROUND',     (0, 0), (0,  -1), colors.HexColor('#eaf0fb')),
    ]))
    story.append(interp_tbl)
    story.append(Spacer(1, 10))

    n_models   = len(top10)
    note_extra = (
        "  Single-predictor runs also test Logarithmic, Exponential, Log-Log, "
        "Reciprocal, Spline, and Lag-1 variants, producing up to 10 candidates.  "
        "Multi-predictor runs fit Linear and Interaction models only, so fewer rows appear."
        if n_models < 10 else ""
    )
    story.append(Paragraph(f"Top {n_models} Models Ranked by BIC", S['h2']))
    story.append(Paragraph(
        "Models ranked by original-scale BIC — lower = better fit penalised for complexity.  "
        "SW p = Shapiro-Wilk (normality);  BP p = Breusch-Pagan (homoscedasticity).  "
        f"Green = PASS (p > 0.05), Red = FAIL (p \u2264 0.05).{note_extra}",
        S['note']))
    story.append(Spacer(1, 4))

    eq_font_base = max(2.5, 6.5 - max(0, len(predictor_cols) - 2) * 0.35)
    model_rows   = [[
        Paragraph('<b>Rank</b>',            S['hdr_cell']),
        Paragraph('<b>Model Type</b>',      S['hdr_cell']),
        Paragraph('<b>Fitted Equation</b>', S['hdr_cell']),
        Paragraph('<b>R²</b>',              S['hdr_cell']),
        Paragraph('<b>Adj R²</b>',          S['hdr_cell']),
        Paragraph('<b>AIC</b>',             S['hdr_cell']),
        Paragraph('<b>BIC</b>',             S['hdr_cell']),
        Paragraph('<b>SW p</b>',            S['hdr_cell']),
        Paragraph('<b>BP p</b>',            S['hdr_cell']),
    ]]

    for i, (_, row) in enumerate(top10.iterrows()):
        model_type_str = str(row.get('ModelType', _model_type_label(int(row['Degree']))))
        eq_font        = 3.25 if 'Spline' in model_type_str else eq_font_base
        sw_hex = PASS_COLOR.hexval()[2:] if row['SW_p'] > 0.05 else FAIL_COLOR.hexval()[2:]
        bp_hex = PASS_COLOR.hexval()[2:] if row['BP_p'] > 0.05 else FAIL_COLOR.hexval()[2:]

        model_rows.append([
            Paragraph(f'<b>{i+1}</b>' if i == 0 else str(i+1),
                      ParagraphStyle('rp', fontSize=8, leading=10,
                                     alignment=TA_CENTER, textColor=DARK)),
            Paragraph(f'<font size="7.5"><b>{model_type_str}</b></font>',
                      ParagraphStyle('tp', fontSize=7.5, leading=10,
                                     alignment=TA_CENTER, textColor=NAVY)),
            Paragraph(f'<font size="{eq_font}">{row["Equation"]}</font>',
                      ParagraphStyle('eq', fontSize=eq_font,
                                     leading=max(4, eq_font * 1.3))),
            Paragraph(str(row['R2']),     S['small']),
            Paragraph(str(row['Adj_R2']), S['small']),
            Paragraph(str(row['AIC']),    S['small']),
            Paragraph(str(row['BIC']),    S['small']),
            Paragraph(f'<font color="#{sw_hex}"><b>{row["SW_p"]}</b></font>', S['small']),
            Paragraph(f'<font color="#{bp_hex}"><b>{row["BP_p"]}</b></font>', S['small']),
        ])

    m_tbl = Table(model_rows,
                  colWidths=[uw*p for p in [0.08, 0.11, 0.37, 0.07, 0.08, 0.08, 0.08, 0.065, 0.065]],
                  repeatRows=1)
    m_style = _tbl_style(NAVY)
    m_style.add('BACKGROUND', (0, 1), (0, 1), colors.HexColor('#fef9e7'))
    m_tbl.setStyle(m_style)
    story.append(m_tbl)
    story.append(Spacer(1, 10))

    # ═══════════════════════════════════════════
    # PAGE — Diagnostic Plots (vertically centred)
    # ═══════════════════════════════════════════
    story.append(PageBreak())

    img_w       = uw
    img_h       = img_w * 0.52
    page_usable = H - 2.2 * cm
    v_pad       = max(0, (page_usable - 20 - 8 - img_h) / 2)

    story.append(Spacer(1, v_pad))
    heading_tbl = Table([[Paragraph("Diagnostic Plots - Best Model", S['h2'])]],
                        colWidths=[uw])
    heading_tbl.setStyle(TableStyle([('ALIGN', (0, 0), (0, 0), 'CENTER')]))
    story.append(heading_tbl)
    story.append(Spacer(1, 8))

    img_tbl = Table([[RLImage(diag_png, width=img_w, height=img_h)]], colWidths=[uw])
    img_tbl.setStyle(TableStyle([
        ('ALIGN',          (0, 0), (0, 0), 'CENTER'),
        ('VALIGN',         (0, 0), (0, 0), 'MIDDLE'),
        ('LEFTPADDING',    (0, 0), (-1,-1), 0),
        ('RIGHTPADDING',   (0, 0), (-1,-1), 0),
        ('TOPPADDING',     (0, 0), (-1,-1), 0),
        ('BOTTOMPADDING',  (0, 0), (-1,-1), 0),
    ]))
    story.append(img_tbl)

    doc.build(story)
    print(f"[PDF] Saved → {out_path}")


# ─────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # ── Configure these for your dataset ──────────────────────────
    CSV_PATH       = 'data.csv'    # path to your CSV file
    TARGET_COL     = 'y'           # dependent variable column name
    PREDICTOR_COLS = ['x']         # list of predictor column names
    OUT_PDF        = f'OLS Regression Analysis {datetime.now().strftime("%d-%m-%Y")}.pdf'
    # ──────────────────────────────────────────────────────────────

    df = load_data(CSV_PATH)

    top10, best_model, best_X, best_degree, best_model_type = select_top_models(
        df, TARGET_COL, PREDICTOR_COLS)

    diag     = run_diagnostics(best_model, best_X)
    ols_png  = make_ols_plot(df, best_model, TARGET_COL, PREDICTOR_COLS,
                              best_degree, diag, model_type=best_model_type)
    diag_png = make_diagnostic_plots(diag)
    corr_png = make_correlation_plot(df, TARGET_COL, PREDICTOR_COLS, model=best_model)

    build_pdf(diag, top10, ols_png, diag_png, OUT_PDF,
              TARGET_COL, PREDICTOR_COLS, best_degree, best_model_type,
              corr_png=corr_png)
