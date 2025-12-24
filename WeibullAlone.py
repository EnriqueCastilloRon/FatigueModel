# Weibull alone
# ========================
# CONFIGURACIÓN INICIAL
# ========================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')
# ========================
# 1. FUNCIONES MATEMÁTICAS BÁSICAS (CORREGIDAS)
# ========================
def compute_V(N, Delta_sigma, logN0, invDelta_sigma0):
    """Calcula la variable transformada V"""
    N0 = np.exp(logN0)
    Delta_sigma0 = 1.0 / invDelta_sigma0
    return np.log(N / N0) * (Delta_sigma / Delta_sigma0)
def pdf_V(v, lambda_, delta, beta):
    """Probability Density Function de V"""
    if v <= lambda_:
        return 0.0
    return weibull_min.pdf(v - lambda_, beta, scale=delta)
def log_likelihood(params, N_data, Delta_sigma_data):
    """
    Función de log-likelihood CORREGIDAS

    La PDF de N dado Δσ es:
    f(N|Δσ) = f_V(v) * |dv/dN|

    donde v = log(N/N₀) * (Δσ/Δσ₀)
    y dv/dN = (Δσ/Δσ₀) * (1/N)
    """
    lambda_, delta, beta, logN0, invDelta_sigma0 = params

    # Verificar positividad de parámetros
    if delta <= 0 or beta <= 0 or invDelta_sigma0 <= 0:
        return -np.inf

    N0 = np.exp(logN0)
    Delta_sigma0 = 1.0 / invDelta_sigma0

    logL = 0.0
    for i in range(len(N_data)):
        N_i = N_data[i]
        Delta_sigma_i = Delta_sigma_data[i]

        # Calcular v
        v = np.log(N_i / N0) * (Delta_sigma_i / Delta_sigma0)

        # Verificar restricción v > lambda
        if v <= lambda_:
            return -np.inf

        # Calcular log-likelihood
        # log f_V(v) para Weibull desplazada
        z = (v - lambda_) / delta
        if z <= 0:
            return -np.inf

        log_pdf_V = np.log(beta) - np.log(delta) + (beta - 1) * np.log(z) - z**beta

        # Jacobiano: |dv/dN| = (Δσ/Δσ₀) / N
        log_jacobian = np.log(Delta_sigma_i / Delta_sigma0) - np.log(N_i)

        logL += log_pdf_V + log_jacobian

    return logL
def percentile_curve(N, p, lambda_, delta, beta, logN0, invDelta_sigma0):
    """Calcula Δσ para un percentil p dado N"""
    N0 = np.exp(logN0)
    Delta_sigma0 = 1.0 / invDelta_sigma0

    v_p = lambda_ + delta * (-np.log(1 - p)) ** (1.0 / beta)

    log_ratio = np.log(N / N0)
    # Evitar división por cero
    log_ratio = np.where(np.abs(log_ratio) < 1e-10, 1e-10, log_ratio)

    return Delta_sigma0 * v_p / log_ratio
# ========================
# 2. OPTIMIZACIÓN MEJORADA
# ========================
def optimize_castillo_canteli(N_data, Delta_sigma_data, opcion):
    """
    Optimización de parámetros del modelo Castillo-Canteli
    opcion=1: Solo optimización global
    opcion=2: Solo optimización local (desde valores iniciales)
    opcion=3: Ambas (global + local)
    """

    print(f"\n{'='*60}")
    print(f"EJECUTANDO OPCIÓN {opcion}")
    print(f"{'='*60}")

    if len(N_data) != len(Delta_sigma_data):
        raise ValueError("N_data y Delta_sigma_data deben tener la misma longitud")

    n = len(N_data)
    print(f"Número de observaciones: {n}")

    # Valores iniciales MEJORADOS basados en los datos
    N_min = np.min(N_data)
    N_max = np.max(N_data)
    Delta_sigma_min = np.min(Delta_sigma_data)
    Delta_sigma_max = np.max(Delta_sigma_data)

    # Estimaciones iniciales más inteligentes, respetando restricciones
    logN0_init = np.log(N_min * 0.1)  # N0 = 10% del mínimo, N0 < N_min
    Delta_sigma0_init = Delta_sigma_min * 0.5  # Δσ₀ = 50% del mínimo, Δσ₀ < Δσ_min
    invDelta_sigma0_init = 1.0 / Delta_sigma0_init

    # Para lambda, estimamos basándonos en el rango de v (usando inits válidos)
    v_estimates = []
    for i in range(len(N_data)):
        v_est = np.log(N_data[i] / np.exp(logN0_init)) * (Delta_sigma_data[i] / Delta_sigma0_init)
        v_estimates.append(v_est)
    v_estimates = np.array(v_estimates)
    min_v_est = np.min(v_estimates)
    max_v_est = np.max(v_estimates)

    lambda_init = min_v_est * 0.5  # Inicial cerca del mínimo v, pero lambda no restringido
    delta_init = (max_v_est - min_v_est) * 0.5
    beta_init = 2.0

    initial_params = [lambda_init, delta_init, beta_init, logN0_init, invDelta_sigma0_init]

    # Bounds ajustados según restricciones
    # lambda: no restringido, pero bounds finitos para DE: amplio alrededor de v
    lambda_lb = min_v_est - 10.0
    lambda_ub = max_v_est + 10.0
    bounds = [
        (lambda_lb, lambda_ub),  # λ: amplio, sin restricción real
        (0.01, 50.0),  # δ > 0, amplio
        (0.2, 20.0),  # β en [0.2, 20]
        (np.log(1e-10), np.log(N_min)),  # logN0: N0 > 0 y N0 < N_min
        (1.0 / Delta_sigma_min, 100.0 / Delta_sigma_min)  # invΔσ₀ > 1/Δσ_min (Δσ₀ < Δσ_min) y permite Δσ₀ pequeño
    ]

    # Imprimir información inicial
    print(f"\nValores iniciales:")
    print(f" λ = {lambda_init:.6f}")
    print(f" δ = {delta_init:.6f}")
    print(f" β = {beta_init:.6f}")
    print(f" logN0 = {logN0_init:.6f} (N₀ = {np.exp(logN0_init):.6f})")
    print(f" invΔσ₀ = {invDelta_sigma0_init:.6f} (Δσ₀ = {1.0/invDelta_sigma0_init:.6f})")

    # Calcular log-likelihood inicial
    initial_loglik = log_likelihood(initial_params, N_data, Delta_sigma_data)
    print(f"\nLog-likelihood inicial: {initial_loglik:.6f}")

    # Función objetivo
    def objective(params):
        ll = log_likelihood(params, N_data, Delta_sigma_data)
        if np.isnan(ll) or np.isinf(ll):
            return 1e10
        return -ll

    # Restricción v_i > λ para todos los datos
    def constraint_v_lambda(params):
        lambda_, delta, beta, logN0, invDelta_sigma0 = params
        N0 = np.exp(logN0)
        Delta_sigma0 = 1.0 / invDelta_sigma0

        min_violation = np.inf
        for i in range(len(N_data)):
            v = np.log(N_data[i] / N0) * (Delta_sigma_data[i] / Delta_sigma0)
            min_violation = min(min_violation, v - lambda_)

        return min_violation

    constraints = [{'type': 'ineq', 'fun': constraint_v_lambda}]

    result_final = None
    x0 = initial_params

    # ETAPA 1: OPTIMIZACIÓN GLOBAL
    if opcion == 1 or opcion == 3:
        print("\n" + "="*50)
        print("ETAPA 1: OPTIMIZACIÓN GLOBAL (Differential Evolution)")
        print("="*50)

        try:
            global_result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=500,
                popsize=30,
                disp=True,
                seed=42,
                tol=1e-8,
                atol=1e-8,
                workers=1
            )

            print(f"\nResultados globales:")
            print(f" λ = {global_result.x[0]:.6f}")
            print(f" δ = {global_result.x[1]:.6f}")
            print(f" β = {global_result.x[2]:.6f}")
            print(f" logN0 = {global_result.x[3]:.6f}")
            print(f" invΔσ₀ = {global_result.x[4]:.6f}")
            print(f" Log-likelihood = {-global_result.fun:.6f}")

            result_final = global_result
            x0 = global_result.x

        except Exception as e:
            print(f"❌ Error en optimización global: {e}")
            print("⚠️ Continuando con valores iniciales...")
            if opcion == 1:
                # Si solo se pidió global y falló, usar valores iniciales
                result_final = type('obj', (object,), {
                    'x': initial_params,
                    'fun': -initial_loglik,
                    'success': False,
                    'message': f'Global optimization failed: {e}'
                })()

    # ETAPA 2: OPTIMIZACIÓN LOCAL
    if opcion == 2 or opcion == 3:
        print("\n" + "="*50)
        print("ETAPA 2: OPTIMIZACIÓN LOCAL (SLSQP)")
        print("="*50)

        try:
            local_result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 5000, 'ftol': 1e-12, 'disp': True}
            )

            print(f"\nResultados locales:")
            print(f" λ = {local_result.x[0]:.6f}")
            print(f" δ = {local_result.x[1]:.6f}")
            print(f" β = {local_result.x[2]:.6f}")
            print(f" logN0 = {local_result.x[3]:.6f}")
            print(f" invΔσ₀ = {local_result.x[4]:.6f}")
            print(f" Log-likelihood = {-local_result.fun:.6f}")

            result_final = local_result

        except Exception as e:
            print(f"❌ Error en optimización local: {e}")
            # Si no hay resultado previo, usar valores iniciales
            if result_final is None:
                print("⚠️ Usando valores iniciales como respaldo...")
                result_final = type('obj', (object,), {
                    'x': initial_params,
                    'fun': -initial_loglik,
                    'success': False,
                    'message': f'Local optimization failed: {e}'
                })()

    # Verificar que tenemos un resultado
    if result_final is None:
        print("❌ ADVERTENCIA: Todas las optimizaciones fallaron. Usando valores iniciales.")
        result_final = type('obj', (object,), {
            'x': initial_params,
            'fun': -initial_loglik,
            'success': False,
            'message': 'All optimizations failed'
        })()

    # RESULTADOS FINALES
    print("\n" + "="*50)
    print("RESULTADOS FINALES")
    print("="*50)

    params_opt = result_final.x
    lambda_opt, delta_opt, beta_opt, logN0_opt, invDelta_sigma0_opt = params_opt

    N0_opt = np.exp(logN0_opt)
    Delta_sigma0_opt = 1.0 / invDelta_sigma0_opt

    print(f"\nParámetros estimados:")
    print(f" λ = {lambda_opt:.6f}")
    print(f" δ = {delta_opt:.6f}")
    print(f" β = {beta_opt:.6f}")
    print(f" N₀ = {N0_opt:.6e}")
    print(f" Δσ₀ = {Delta_sigma0_opt:.6f}")
    print(f" Log-likelihood = {-result_final.fun:.6f}")

    # Verificar restricciones
    print(f"\nVerificación de restricciones:")
    print(f" N₀ < min(N): {N0_opt < N_min}")
    print(f" Δσ₀ < min(Δσ): {Delta_sigma0_opt < Delta_sigma_min}")
    print(f" δ > 0: {delta_opt > 0}")
    print(f" β ∈ [0.2, 20]: {0.2 <= beta_opt <= 20}")
    violations = []
    for i in range(len(N_data)):
        v = np.log(N_data[i] / N0_opt) * (Delta_sigma_data[i] / Delta_sigma0_opt)
        violations.append(v - lambda_opt)

    min_violation = np.min(violations)
    print(f" Min(v_i - λ) = {min_violation:.6f} (debe ser > 0)")
    print(f" Todos los puntos satisfacen v_i > λ: {min_violation > 0}")

    results = {
        'parameters_original': {
            'lambda': lambda_opt,
            'delta': delta_opt,
            'beta': beta_opt,
            'N0': N0_opt,
            'Delta_sigma0': Delta_sigma0_opt,
            'log_likelihood': -result_final.fun
        },
        'success': getattr(result_final, 'success', False),
        'message': getattr(result_final, 'message', 'N/A'),
        'n_observations': n,
        'opcion_elegida': opcion
    }

    return results
# ========================
# 3. VISUALIZACIÓN
# ========================
def plot_results(N_data, Delta_sigma_data, params_original, percentiles=None, opcion=3):
    """Grafica los datos y las curvas percentiles del modelo"""

    if percentiles is None:
        percentiles = [0.01, 0.10, 0.50, 0.90, 0.99]

    lambda_ = params_original['lambda']
    delta = params_original['delta']
    beta = params_original['beta']
    N0 = params_original['N0']
    Delta_sigma0 = params_original['Delta_sigma0']

    logN0 = np.log(N0)
    invDelta_sigma0 = 1.0 / Delta_sigma0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. GRÁFICO PRINCIPAL
    ax1 = axes[0, 0]
    ax1.scatter(N_data, Delta_sigma_data, alpha=0.6, s=50, label='Datos', color='blue', zorder=5)

    # Rango ajustado: ligeramente ampliado desde los datos, evitando N < N0
    N_min_data = np.min(N_data)
    N_max_data = np.max(N_data)
    N_start = np.max([N0 * 1.1, N_min_data * 0.8])
    N_end = N_max_data * 1.5
    N_range = np.logspace(np.log10(N_start), np.log10(N_end), 300)

    colors = plt.cm.viridis(np.linspace(0, 1, len(percentiles)))

    for p, color in zip(percentiles, colors):
        try:
            Delta_sigma_curve = percentile_curve(N_range, p, lambda_, delta, beta,
                                               logN0, invDelta_sigma0)
            # Filtrar valores no físicos
            valid = (Delta_sigma_curve > 0) & (Delta_sigma_curve < 10)
            ax1.plot(N_range[valid], Delta_sigma_curve[valid], color=color,
                    label=f'p = {p:.2f}', linewidth=2.5, alpha=0.8)
        except:
            continue

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('N (ciclos)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Δσ (rango de tensión)', fontsize=12, fontweight='bold')

    opcion_text = {1: "Solo global", 2: "Solo local", 3: "Global + local"}
    ax1.set_title(f'Curvas p-S-N del Modelo Castillo-Canteli\n({opcion_text[opcion]})',
                  fontsize=14, fontweight='bold')

    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3, which='both')

    # 2. DISTRIBUCIÓN DE V
    ax2 = axes[0, 1]

    v_data = []
    for i in range(len(N_data)):
        v = compute_V(N_data[i], Delta_sigma_data[i], logN0, invDelta_sigma0)
        v_data.append(v)

    v_data = np.array(v_data)

    ax2.hist(v_data, bins=20, density=True, alpha=0.6, color='green',
             edgecolor='black', label='Datos V')

    v_min_plot = min(lambda_, np.min(v_data) - 1)
    v_range = np.linspace(v_min_plot, max(v_data) * 1.2, 200)
    pdf_values = [pdf_V(v, lambda_, delta, beta) for v in v_range]
    ax2.plot(v_range, pdf_values, 'r-', linewidth=2.5, label='PDF Weibull ajustada')

    ax2.axvline(lambda_, color='orange', linestyle='--', linewidth=2, label=f'λ = {lambda_:.2f}')

    ax2.set_xlabel('V (variable transformada)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Densidad', fontsize=12, fontweight='bold')
    ax2.set_title('Distribución de V', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. QQ-PLOT
    ax3 = axes[1, 0]

    n = len(v_data)
    theoretical_quantiles = []
    empirical_quantiles = np.sort(v_data)

    for i in range(1, n + 1):
        p = i / (n + 1)
        v_p = lambda_ + delta * (-np.log(1 - p)) ** (1.0 / beta)
        theoretical_quantiles.append(v_p)

    theoretical_quantiles = np.array(theoretical_quantiles)

    ax3.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.7, s=40, color='purple')

    min_val = min(theoretical_quantiles.min(), empirical_quantiles.min())
    max_val = max(theoretical_quantiles.max(), empirical_quantiles.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

    # Calcular R²
    residuals = empirical_quantiles - theoretical_quantiles
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((empirical_quantiles - np.mean(empirical_quantiles))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    ax3.set_xlabel('Cuantiles Teóricos', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cuantiles Empíricos', fontsize=12, fontweight='bold')
    ax3.set_title(f'QQ-Plot (R² = {r2:.4f})', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. INFORMACIÓN
    ax4 = axes[1, 1]
    ax4.axis('off')

    opcion_detalle = {1: "Solo global (DE)", 2: "Solo local (SLSQP)", 3: "Global + local"}

    info_text = (
        f"PARÁMETROS OPTIMIZADOS:\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"λ = {lambda_:.4f}\n"
        f"δ = {delta:.4f}\n"
        f"β = {beta:.4f}\n"
        f"N₀ = {N0:.4e}\n"
        f"Δσ₀ = {Delta_sigma0:.4f}\n\n"
        f"ESTADÍSTICAS:\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Log-likelihood = {params_original.get('log_likelihood', 'N/A'):.2f}\n"
        f"N observaciones = {len(N_data)}\n"
        f"R² (QQ-plot) = {r2:.4f}\n"
        f"Opción: {opcion_detalle[opcion]}"
    )

    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    plt.show()

    return fig
# ========================
# 4. MENÚ
# ========================
def mostrar_menu():
    print("\n" + "="*60)
    print("OPCIONES DE OPTIMIZACIÓN")
    print("="*60)
    print("1. Solo GLOBAL (Differential Evolution)")
    print("2. Solo LOCAL (SLSQP)")
    print("3. AMBAS: Global + Local (⭐ RECOMENDADO)")
    print("="*60)
def seleccionar_opcion():
    while True:
        try:
            opcion = int(input("\nSeleccione (1-3): "))
            if opcion in [1, 2, 3]:
                return opcion
            print("❌ Por favor, seleccione 1, 2 o 3")
        except ValueError:
            print("❌ Ingrese un número válido")
# ========================
# 5. EJECUCIÓN PRINCIPAL
# ========================
def main():
    Delta_sigma_data = np.array([
        0.950,0.950,0.950,0.950,0.950,0.950,0.950,0.950,0.950,0.950,0.950,0.950,0.950,0.950,0.950,
        0.900,0.900,0.900,0.900,0.900,0.900,0.900,0.900,0.900,0.900,0.900,0.900,0.900,0.900,0.900,
        0.825,0.825,0.825,0.825,0.825,0.825,0.825,0.825,0.825,0.825,0.825,0.825,0.825,0.825,0.825,
        0.750,0.750,0.750,0.750,0.750,0.750,0.750,0.750,0.750,0.750,0.750,0.750,0.750,0.750,0.750,
        0.675,0.675,0.675,0.675,0.675,0.675,0.675,0.675,0.675,0.675,0.675,0.675,0.675,0.675,0.675
    ])

    N_data = np.array([
        37,72,74,76,83,85,105,109,120,123,143,203,206,217,257,
        201,216,226,252,257,295,311,342,356,451,457,509,540,680,1129,
        1246,1258,1460,1492,2400,2410,2590,2903,3330,3590,3847,4110,4820,5560,5598,
        6710,9930,12600,15580,16190,17280,18620,20300,24900,26260,27940,36350,48420,50090,67340,
        102950,280320,339830,366900,485620,658960,896330,1241760,1250200,1329780,1399830,1459140,3294820,12709000,14373000
    ])

    percentiles = [0.01, 0.10, 0.50, 0.90, 0.99]

    print("="*60)
    print("MODELO CASTILLO-CANTELI PARA CURVAS p-S-N")
    print("="*60)
    print(f"\nDatos: {len(N_data)} observaciones")
    print(f"N ∈ [{np.min(N_data):.1f}, {np.max(N_data):.1e}]")
    print(f"Δσ ∈ [{np.min(Delta_sigma_data):.3f}, {np.max(Delta_sigma_data):.3f}]")

    mostrar_menu()
    opcion = seleccionar_opcion()

    results = optimize_castillo_canteli(N_data, Delta_sigma_data, opcion)

    print("\n" + "="*60)
    print("GENERANDO GRÁFICOS...")
    print("="*60)

    fig = plot_results(N_data, Delta_sigma_data,
                      results['parameters_original'],
                      percentiles=percentiles,
                      opcion=opcion)

    return results
if __name__ == "__main__":
    results = main()

    print("\n" + "="*60)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*60)
    print(f"Log-likelihood: {results['parameters_original']['log_likelihood']:.2f}")
    print("="*60)