import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.signal import welch
from scipy.stats import linregress

# ============================================================
# PARÂMETROS GERAIS
# ============================================================

sigma = 0.4
a = 1.0
b0 = 2.0
kappa = 1.0
dt = 0.01
xi_c = b0 / kappa

# ============================================================
# POTENCIAL E DRIFT
# ============================================================

def b_xi(xi):
    return b0 - kappa * xi

def dU_dx(x, xi):
    b = b_xi(xi)
    return a*x**3 - b*x

# ============================================================
# SIMULAÇÃO 1D
# ============================================================

def simulate_1d(xi, T, seed=None):
    if seed:
        np.random.seed(seed)
    x = 0.1
    xs = []
    for _ in range(T):
        drift = -dU_dx(x, xi)
        noise = sigma*np.sqrt(dt)*np.random.randn()
        x += drift*dt + noise
        xs.append(x)
    return np.array(xs)

# ============================================================
# SWEEP + SUSCETIBILIDADE
# ============================================================

xis = np.linspace(0.2, 3, 40)
vars_ = []

for xi in xis:
    traj = simulate_1d(xi, 20000)
    vars_.append(np.var(traj))

vars_ = np.array(vars_)
chi = np.gradient(vars_, xis)

plt.figure()
plt.plot(xis, vars_)
plt.title("Variância vs ξ")
plt.axvline(xi_c, color='r', linestyle='--')
plt.show()

plt.figure()
plt.plot(xis, chi)
plt.title("Suscetibilidade χ")
plt.axvline(xi_c, color='r', linestyle='--')
plt.show()

# ============================================================
# EXPOENTE CRÍTICO
# ============================================================

distance = np.abs(xis - xi_c) + 1e-3
logx = np.log(distance)
logy = np.log(vars_)
coef = np.polyfit(logx, logy, 1)
gamma_est = -coef[0]

plt.figure()
plt.plot(logx, logy)
plt.title(f"Expoente crítico γ ≈ {gamma_est:.2f}")
plt.show()

# ============================================================
# TEMPO DE ESCAPE (FIRST PASSAGE)
# ============================================================

def mean_escape_time(xi, T):
    traj = simulate_1d(xi, T)
    saddle = 0.0
    crossings = np.where(np.diff(np.sign(traj - saddle)))[0]
    if len(crossings) < 2:
        return np.nan
    intervals = np.diff(crossings)
    return np.mean(intervals)*dt

escape_times = [mean_escape_time(xi, 30000) for xi in xis]

plt.figure()
plt.plot(xis, escape_times)
plt.title("Tempo Médio de Escape")
plt.show()

# ============================================================
# SLOWING DOWN (AUTOCORR)
# ============================================================

def lag1_autocorr(ts):
    return np.corrcoef(ts[:-1], ts[1:])[0,1]

autocorrs = []
for xi in xis:
    traj = simulate_1d(xi, 10000)
    autocorrs.append(lag1_autocorr(traj))

plt.figure()
plt.plot(xis, autocorrs)
plt.title("Critical Slowing Down")
plt.show()

# ============================================================
# ESPECTRO DE POTÊNCIA
# ============================================================

traj = simulate_1d(1.9, 40000)
f, Pxx = welch(traj, fs=1/dt)

plt.figure()
plt.loglog(f[1:], Pxx[1:])
plt.title("Espectro de Potência")
plt.show()

# ============================================================
# MODELO 2D
# ============================================================

def simulate_2d(xi, T, coupling=0.5):
    x, y = 0.1, -0.1
    xs, ys = [], []
    for _ in range(T):
        dx = -dU_dx(x, xi) + coupling*(y-x)
        dy = -dU_dx(y, xi) + coupling*(x-y)
        x += dx*dt + sigma*np.sqrt(dt)*np.random.randn()
        y += dy*dt + sigma*np.sqrt(dt)*np.random.randn()
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

x2, y2 = simulate_2d(1.5, 20000)

plt.figure()
plt.plot(x2, label="Ativo 1")
plt.plot(y2, label="Ativo 2")
plt.legend()
plt.show()

print("Correlação 2D:", np.corrcoef(x2,y2)[0,1])

# ============================================================
# MATRIZ DE DIFUSÃO
# ============================================================

dx = np.diff(x2)
dy = np.diff(y2)

Dxx = np.var(dx)/dt
Dyy = np.var(dy)/dt
Dxy = np.cov(dx,dy)[0,1]/dt

D = np.array([[Dxx, Dxy],[Dxy,Dyy]])
print("Matriz de Difusão:\n", D)

# ============================================================
# HURST
# ============================================================

def hurst(ts):
    N = len(ts)
    T_vals = np.arange(20, N//2, 200)
    RS = []
    for T in T_vals:
        n = N//T
        rs_vals = []
        for i in range(n):
            seg = ts[i*T:(i+1)*T]
            Z = np.cumsum(seg-np.mean(seg))
            R = np.max(Z)-np.min(Z)
            S = np.std(seg)
            if S>0:
                rs_vals.append(R/S)
        if rs_vals:
            RS.append(np.mean(rs_vals))
    slope,_,_,_,_ = linregress(np.log(T_vals[:len(RS)]), np.log(RS))
    return slope

print("Hurst:", hurst(x2))

# ============================================================
# ISING MEAN FIELD
# ============================================================

def simulate_ising(N=200, J=1.0, beta=1.2, steps=300):
    spins = np.random.choice([-1,1], N)
    mags = []
    for _ in range(steps):
        m = np.mean(spins)
        for i in range(N):
            h = J*m
            p = 1/(1+np.exp(-2*beta*h))
            spins[i] = 1 if np.random.rand()<p else -1
        mags.append(np.mean(spins))
    return np.array(mags)

mags = simulate_ising()
plt.figure()
plt.plot(mags)
plt.title("Magnetização Ising")
plt.show()

# ============================================================
# CONTROLE ADAPTATIVO
# ============================================================

def adaptive_xi(ts, base=1.0):
    return base/(1+np.var(ts))

print("xi adaptativo:", adaptive_xi(x2))

# ============================================================
# APLICAÇÃO A DADOS REAIS (S&P 500)
# ============================================================

data = yf.download("^GSPC", start="2000-01-01", end="2012-01-01")
returns = np.diff(np.log(data['Adj Close'].values))

plt.figure()
plt.plot(returns)
plt.title("Retornos S&P 500")
plt.show()

# Variância móvel
window = 250
rolling_var = [np.var(returns[i-window:i]) 
               for i in range(window,len(returns))]

plt.figure()
plt.plot(rolling_var)
plt.title("Variância móvel (250 dias)")
plt.show()

# Autocorrelação móvel
rolling_ac = [np.corrcoef(returns[i-window:i-1],
                          returns[i-window+1:i])[0,1]
              for i in range(window,len(returns))]

plt.figure()
plt.plot(rolling_ac)
plt.title("Autocorrelação lag-1 móvel")
plt.show()

# Reconstrução de potencial empírico
hist, bins = np.histogram(returns, bins=100, density=True)
centers = (bins[:-1]+bins[1:])/2
p = hist+1e-8
U_est = -np.log(p)

plt.figure()
plt.plot(centers, U_est)
plt.title("Potencial Efetivo Empírico")
plt.show()
