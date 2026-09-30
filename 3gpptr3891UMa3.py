import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

rng = np.random.default_rng()


def distancia_calc(x_tx, y_tx, h_tx, x_rx, y_rx, h_rx):
    d_x = x_rx - x_tx
    d_y = y_rx - y_tx
    d_z = h_rx - h_tx

    d_2d = ( d_y**2  + d_x**2 )**0.5# min35m
    d_3d = ( d_2d**2 + d_z**2 )**0.5    

    return d_x, d_y, d_z, d_2d, d_3d

def angulo_calc(d_x, d_y, d_z, d_2d):
    phi_o = np.arctan2(d_y, d_x)
    phi_i = np.arctan2(-d_y, -d_x)

    theta_o = np.arctan2(d_z, d_2d)
    theta_i = np.arctan2(-d_z, d_2d)

    return phi_o, phi_i, theta_o, theta_i

def prob_los(d_2d, h_rx):
    C_h_rx = 0
    if h_rx > 13:
        C_h_rx = ((h_rx - 13)/10)**1.5

    p_LoS = 1
    if d_2d > 18:
        p_LoS = ( 
            18/d_2d
            + np.exp(-d_2d/63)*(1-18/d_2d)
        )*(
            1
            + C_h_rx*(5/4)
            * (d_2d/100)**3
            *np.exp(-d_2d/150)
        )

    return p_LoS


def parametros_esp_atraso(LoS, fc):
    
    if LoS:
        return -6.955 -0.0963*np.log10(fc), 0.66
    else:
        return -6.28 -0.204*np.log10(fc), 0.39



def parametros_esp_az_o(LoS, fc):
    if LoS:
        return 1.06 + 0.1114*np.log10(fc), 0.28
    else:
        return 1.5 - 0.1144*np.log10(fc), 0.28


def parametros_esp_az_i(LoS, fc):
    if LoS:
        return 1.81, 0.20
    else:
        return 2.08 - 0.27*np.log10(fc), 0.11


def parametros_esp_zen_o(LoS, d_2d, h_ut):
    if LoS:
        return max(-0.5, -2.1*(d_2d/1000) -0.01*(h_ut-1.5)+0.75), 0.40
    else:
        return max(-0.5, -2.1*(d_2d/1000) -0.01*(h_ut-1.5)+0.9), 0.49


def parametros_esp_zen_i(LoS, fc):
    if LoS:
        return 0.95, 0.16
    else:
        return -0.3236*np.log10(fc) + 1.512, 0.16

def parametros_larga_escala(LoS, fc, d_2d, h_ut, rng):

    fc_lsp = max(fc, 6.0) #Nota 6

    #Espalhamento de atraso
    mu, sigma = parametros_esp_atraso(LoS, fc_lsp)
    ds = 10**rng.normal(mu, sigma)

    #Espalhamento azimutal saida
    mu, sigma = parametros_esp_az_o(LoS, fc_lsp)
    asd = min(10**rng.normal(mu, sigma), 104.0)

    #Espalhamento azimutal chegada
    mu, sigma = parametros_esp_az_i(LoS, fc_lsp)
    asa = min(10**rng.normal(mu, sigma), 104.0)

    #Espalhamento zenit saida
    mu, sigma = parametros_esp_zen_o(LoS, d_2d, h_ut)
    zsd = min(10**rng.normal(mu, sigma), 52.0)

    #Espalhamento zenit chegada
    mu, sigma = parametros_esp_zen_i(LoS, fc_lsp)
    zsa = min(10**rng.normal(mu, sigma), 52.0)

    #rice
    if LoS:
        mu, sigma = 9, 3.5
        k = 10**(rng.normal(mu, sigma)/10.0)
    else:
        k = 0

    return {"ds": ds, "asd": asd, "asa": asa, "zsd": zsd, "zsa": zsa, "k": k}



def atrasos_multipercurso(N, r_tau, sigma_tau, rng):
    mu_tau = r_tau*sigma_tau
    atrasos = np.zeros(N)

    for n in range(N):
        atrasos[n] = rng.exponential(mu_tau)

    atrasos = atrasos - np.min(atrasos)
    atrasos = np.sort(atrasos)

    return atrasos


def perf_potencia_multipercurso(tau, r_tau, sigma_tau, csi):
    return np.exp(-tau*(r_tau-1)/(r_tau*sigma_tau))*10**(-csi/10)

def potencias_multipercurso(LoS, N, k_r, tau, r_tau, sigma_tau, sigma_csi, rng):

    potencia_pre = np.zeros(N)

    for n in range(N):
        csi = rng.normal(0, sigma_csi)
        potencia_pre[n] = perf_potencia_multipercurso(tau[n], r_tau, sigma_tau, csi)

    if LoS:
        omega_c = np.sum(potencia_pre[1:])

        potencias = np.zeros(N)
        potencias[1:] = potencia_pre[1:] / (omega_c *(k_r+1))
        potencias[0] = k_r/(k_r+1)

    else:
        omega_c = np.sum(potencia_pre)
      
        potencias = potencia_pre/omega_c

    return potencias


def modelo_azimut(asa, a2n, a2_max):
    return 1.42*asa*np.sqrt(-np.log(a2n/a2_max))

def angulos_azimut(LoS, N, asa, a2, phi_i, rng):

    phi_i_graus = np.degrees(phi_i)
    angulos_i = np.zeros(N)
    a2_max = np.max(a2)

    for n in range(N):
        angulos_i[n] = modelo_azimut(asa, a2[n], a2_max)

    sinais = rng.choice([-1,1], size=N)
    Yn = rng.normal(0, asa/7, size=N)


    angulos = sinais*angulos_i + Yn + phi_i_graus

    if LoS:
        angulos[0] = phi_i_graus

    angulos = (angulos + 180) % 360 - 180
    return angulos


def modelo_zenit(zsa, a2n, a2_max):
    return -zsa*np.log(a2n/a2_max)

def angulos_zenit(LoS, N, zsa, a2, theta_i, rng):
    theta_i_graus = np.degrees(theta_i)
    angulos_i = np.zeros(N)
    a2_max = np.max(a2)

    for n in range(N):
        angulos_i[n] = modelo_zenit(zsa, a2[n], a2_max)

    sinais = rng.choice([-1,1], size=N)
    Yn = rng.normal(0, zsa/7, size=N)

    angulos = sinais*angulos_i + Yn + theta_i_graus

    if LoS:
        angulos[0] = theta_i_graus

    angulos = angulos%360


    return angulos


def direcoes_chegada(N, phi_chegada, theta_chegada):
    phi = np.radians(phi_chegada)
    theta = np.radians(theta_chegada)
    vetores = np.zeros((N,3))

    for n in range(N):
        vetores[n, 0] = np.cos(phi[n])*np.sin(theta[n])
        vetores[n, 1] = np.sin(phi[n])*np.sin(theta[n])
        vetores[n, 2] = np.cos(theta[n])
        #vetores[n, 0] = np.cos(phi[n])*np.cos(theta[n])
        #vetores[n, 1] = np.sin(phi[n])*np.cos(theta[n])
        #vetores[n, 2] = np.sin(theta[n])


    return vetores

def desvios_doppler(N, r, v_rx, v_rx_unit, fc):
    c = 3e8
    fc_hz = fc*1e9
    comprimento = c/fc_hz

    desvios = np.zeros(N)

    for n in range(N):
        produto = np.dot(r[n], v_rx_unit)
        desvios[n] = (v_rx/comprimento)*produto

    return desvios

def fases_multipercurso(N, fc, vu, tau, t):

    fc_hz = fc*1e9
    fases_barra = np.zeros(N)
    fases = np.zeros(N)

    for n in range(N):
        fases_barra[n] = 2*np.pi*((fc_hz + vu[n])*tau[n])
        fases[n] = fases_barra[n] - 2*np.pi*vu[n]*t

    return fases

def pulso(atraso, delta_t, Nt):
    t = np.linspace(0, 5*delta_t, Nt)
    sinal = np.zeros(Nt)

    for i in range(Nt):
        if (t[i] >= atraso) and (t[i] < atraso + delta_t):
            sinal[i] = 1

    return sinal

def sinal_recebido(N, alpha2, tau, vu, fc, delta_t, Nt):
    fc_hz = fc *1e9
    t = np.linspace(0, 5*delta_t, Nt)
    recebido = np.zeros(Nt, dtype=complex)

    for n in range(N):
        recebido = recebido + np.sqrt(alpha2[n])*np.exp(-2*np.pi*1j*((fc_hz+vu[n])*tau[n]-vu[n]*t))*pulso(tau[n], delta_t, Nt)

    return recebido


def autocor_normalizada(a2, tau, vu, kappa, sigma):
    omega_c = np.sum(a2)

    soma = 0
    for n in range(len(a2)):
        soma = soma + a2[n]*np.exp(-1j*2*np.pi*tau[n]*kappa)*np.exp(1j*2*np.pi*vu[n]*sigma)

    return soma/omega_c
#Sistema Cartesiano

#Definicao de parametros
f_c = 3 #GHz


#   Base Station
h_tx = 25 #tabela 7.2-1
y_tx = 0
x_tx = 0

#   User Terminal
y_rx = 30
x_rx = 40

n_fl = 5 #andar (nivel do solo = 1)
h_rx = 3*(n_fl -1) + 1.5 #3D-UMa em TR36.873

v_rx_kmh = 3
v_rx = v_rx_kmh / 3.6
v_rx_unit = np.array([1, 0, 0])




#==================================================

#   Distancias
d_x, d_y, d_z, d_txrx_2d, d_txrx_3d = distancia_calc(x_tx, y_tx, h_tx, x_rx, y_rx, h_rx)

if d_txrx_2d < 35:
    raise ValueError("A distância 2D deve ser pelo menos 35 m.")


#   LoS
phi_o, phi_i, theta_o, theta_i = angulo_calc(d_x, d_y, d_z, d_txrx_2d)


#p_LoS = 
p_LoS = prob_los(d_txrx_2d, h_rx)

LoS = rng.binomial(n=1, p=p_LoS)





print(f"Distância 2D: {d_txrx_2d:.2f} m")
print(f"Distância 3D: {d_txrx_3d:.2f} m")

print(f"Azimute de saída: {np.degrees(phi_o):.2f}°")
print(f"Azimute de chegada: {np.degrees(phi_i):.2f}°")
print(f"Zenital de saída: {np.degrees(theta_o):.2f}°")
print(f"Zenital de chegada: {np.degrees(theta_i):.2f}°")

print(f"Probabilidade de LoS: {p_LoS:.2%}")

#   Parametros Larga Escala
lsp = parametros_larga_escala(LoS, f_c, d_txrx_2d, h_rx, rng)

print(f"Estado do enlace: {'LoS' if LoS else 'NLoS'}")
print(f"DS: {lsp['ds']:.3e} s")
print(f"ASD: {lsp['asd']:.2f}°")
print(f"ASA: {lsp['asa']:.2f}°")
print(f"ZSD: {lsp['zsd']:.2f}°")
print(f"ZSA: {lsp['zsa']:.2f}°")
print(f"K linear: {lsp['k']:.3f}")


# Atraso Multipercurso

#tabela 7.5-6
#N = 12 if LoS else 20 #numero de multipercursos
N = 100
r_tau = 2.5 if LoS else 2.3 #

tau = atrasos_multipercurso(N, r_tau, lsp['ds'], rng)

# Potencia Multipercurso

#tabela 7.5-6
sigma_csi = 3

alpha2 = potencias_multipercurso(LoS, N, lsp['k'], tau, r_tau, lsp['ds'], sigma_csi, rng)




#   Direcoes de Chegada

#Azimutal
phi_chegada = angulos_azimut(LoS, N, lsp['asa'], alpha2, phi_i, rng)

#em Elevacao
theta_chegada = angulos_zenit(LoS, N, lsp['zsa'], alpha2, theta_i, rng)

#Vetores
r_unit = direcoes_chegada(N, phi_chegada, theta_chegada)


#   Doppler
vu = desvios_doppler(N, r_unit, v_rx, v_rx_unit, f_c)

#Fases Multipercurso
t = 0
fases = fases_multipercurso(N, f_c, vu, tau, t)


#   Espalhamento Temporal
Nt = 10**5
delta_t = 1e-7

sinal_tx = pulso(0, delta_t, Nt)
sinal_rx = sinal_recebido(N, alpha2, tau, vu, f_c, delta_t, Nt)


#   Banda e Tempo de Coerencia
rho_B = 0.8
rho_T = 0.8

kappa = np.linspace(0, 100e6, 10001)
sigma = np.linspace(0, 1.0, 10001)

#banda
B_c = None
rho_f = np.abs(autocor_normalizada(alpha2, tau, vu, kappa=kappa, sigma=0.0))

for i in range(len(kappa)):
    if rho_f[i] < rho_B:
        B_c = kappa[i]
        break

#tempo
T_c = None
rho_t = np.abs(autocor_normalizada(alpha2, tau, vu, kappa=0.0, sigma=sigma))

for i in range(len(sigma)):
    if rho_t[i] < rho_T:
        T_c = sigma[i]
        break






from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

estado = "LoS" if LoS else "NLoS"
contexto = f"UMa | {estado} | {f_c:g} GHz | N = {N} | v = {v_rx:.3f} m/s"
azul, laranja, verde = "#2563a6", "#db650c", "#16806a"
piso_db = -50.0

# Potencias relativas ao pico: a mesma referencia em todos os perfis.
# O piso afeta somente a visualizacao, nunca alpha2.
potencia_db = 10 * np.log10(np.maximum(
    alpha2 / np.max(alpha2), 10.0**(piso_db / 10)
))
cores = [laranja if LoS and n == 0 else azul for n in range(N)]

# Recalcula as primeiras amostras abaixo do limiar para as anotacoes.
indices_b = np.flatnonzero(rho_f < rho_B)
indices_t = np.flatnonzero(rho_t < rho_T)
B_c = kappa[indices_b[0]] if indices_b.size else None
T_c = sigma[indices_t[0]] if indices_t.size else None


def grade_log(ax, eixo="y"):
    axis = ax.yaxis if eixo == "y" else ax.xaxis
    axis.set_major_locator(LogLocator(base=10))
    axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    axis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", alpha=0.5, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.22, linewidth=0.5, linestyle=":")


def hastes_db(ax, x, titulo, unidade):
    piso = 10.0**(piso_db / 10)
    potencia_rel = np.maximum(alpha2 / np.max(alpha2), piso)
    ax.vlines(x, piso, potencia_rel, colors=cores, linewidth=1.2)
    ax.scatter(x, potencia_rel, c=cores, s=24, zorder=3)
    ax.set_yscale("log")
    ax.set(title=titulo, xlabel=unidade,
           ylabel="Potência / pico (escala log)", ylim=(piso, 1.5))
    grade_log(ax)


def curva_coerencia(ax, separacao, correlacao, limiar, cruzamento,
                    titulo, unidade):
    positivos = separacao > 0
    ax.plot(separacao[positivos], correlacao[positivos], color=verde, linewidth=1.1)
    ax.set_xscale("log")
    ax.axhline(limiar, color=laranja, linestyle="--",
               label=f"Limiar |ρ| = {limiar:g}")
    if cruzamento is not None:
        ax.axvline(cruzamento, color="#333333", linestyle=":",
                   label=f"Primeiro cruzamento ≈ {cruzamento:.3g} {unidade}")
    else:
        ax.text(0.98, 0.06, "Sem cruzamento na faixa analisada",
                ha="right", transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    ax.set(title=titulo, xlabel=f"Separação ({unidade}, escala log)",
           ylabel="Módulo da correlação |ρ|", ylim=(0, 1.05),
           xlim=(separacao[positivos][0], separacao[-1]))
    grade_log(ax, eixo="x")
    ax.legend(loc="upper right", fontsize=8)


# JANELA 1: cada perfil fica acima da correlacao correspondente.
fig_perfis, eixos = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
fig_perfis.suptitle("Potência e coerência do canal\n" + contexto)
hastes_db(eixos[0, 0], tau * 1e9, "Perfil de potência por atraso", "Atraso relativo (ns)")
hastes_db(eixos[0, 1], vu, "Espectro Doppler discreto", "Desvio Doppler (Hz)")
curva_coerencia(eixos[1, 0], kappa / 1e6, rho_f, rho_B,
                None if B_c is None else B_c / 1e6,
                "Banda de coerência", "MHz")
curva_coerencia(eixos[1, 1], sigma * 1e3, rho_t, rho_T,
                None if T_c is None else T_c * 1e3,
                "Tempo de coerência", "ms")
if LoS:
    eixos[0, 0].plot([], [], "o", color=laranja, label="Componente direta")
    eixos[0, 0].legend(fontsize=8)


def polar_db(ax, angulos_graus, titulo):
    raios = potencia_db - piso_db

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetamin(0)
    ax.set_thetamax(360)
    ax.set_thetagrids(np.arange(0, 360, 45))

    for n in range(N):
        angulo = np.radians(angulos_graus[n] % 360)

        ax.plot(
            [angulo, angulo], [0, raios[n]],
            color=cores[n], linewidth=1
        )
        ax.plot(
            angulo, raios[n],
            "o", color=cores[n], markersize=3
        )

    # Anéis principais em dB
    niveis = np.arange(piso_db, 1, 10)
    ax.set_yticks(niveis - piso_db)
    ax.set_yticklabels(
        [f"{nivel:g}" for nivel in niveis],
        fontsize=8
    )
    ax.set_ylim(0, -piso_db)
    ax.set_rlabel_position(135)

    # Subdivisões logarítmicas da potência
    raios_secundarios = []

    for expoente in range(int(piso_db / 10), 0):
        for fator in range(2, 10):
            raio = (
                10 * np.log10(fator * 10.0**expoente)
                - piso_db
            )
            raios_secundarios.append(raio)

    ax.set_yticks(raios_secundarios, minor=True)
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.grid(True, which="major", alpha=0.5, linewidth=0.8)
    ax.grid(
        True, which="minor",
        alpha=0.22, linewidth=0.5, linestyle=":"
    )

    ax.set_title(
        titulo + "\nPotência relativa ao pico (dB)",
        pad=22
    )


# JANELA 2: ângulos e vetores de direção de chegada
fig_direcoes = plt.figure(figsize=(15, 6.5))
fig_direcoes.subplots_adjust(
    left=0.04, right=0.93,
    bottom=0.12, top=0.76,
    wspace=0.32
)
fig_direcoes.suptitle("Direções de chegada\n" + contexto)

ax_az = fig_direcoes.add_subplot(131, projection="polar")
ax_el = fig_direcoes.add_subplot(132, projection="polar")
ax_3d = fig_direcoes.add_subplot(133, projection="3d")

polar_db(ax_az, phi_chegada, "Azimute")
polar_db(ax_el, theta_chegada, "Elevação — representação circular")

for n in range(N):
    x, y, z = r_unit[n]

    ax_3d.quiver(
        0, 0, 0,
        x, y, z,
        color=cores[n],
        arrow_length_ratio=0.12,
        linewidth=1.1
    )

ax_3d.scatter(0, 0, 0, color="black", s=20)

ax_3d.set(
    xlabel="X",
    ylabel="Y",
    zlabel="Z",
    xlim=(-1, 1),
    ylim=(-1, 1),
    zlim=(-1, 1),
    title="Vetores unitários\nComprimento não representa potência"
)

ax_3d.set_xticks([-1, 0, 1])
ax_3d.set_yticks([-1, 0, 1])
ax_3d.set_zticks([-1, 0, 1])
ax_3d.set_box_aspect((1, 1, 1))
ax_3d.view_init(elev=25, azim=-60)

if LoS:
    ax_3d.legend(
        handles=[
            Line2D(
                [0], [0],
                color=laranja,
                label="Componente direta"
            )
        ],
        loc="lower center",
        fontsize=8
    )

# JANELA 3: amplitude linear e potencia em eixo logaritmico.
# Usa exatamente a janela que gerou sinal_tx e sinal_rx.
t_plot = np.linspace(0, 5 * delta_t, Nt)
t_ns = t_plot * 1e9
fig_sinais, (ax_linear, ax_db) = plt.subplots(
    2, 1, figsize=(11, 7), sharex=True, layout="constrained"
)
fig_sinais.suptitle(
    "Espalhamento temporal do sinal\n" + contexto
    + f" | pulso = {delta_t * 1e9:.1f} ns | DS sorteado = {lsp['ds'] * 1e9:.1f} ns"
)
ax_linear.plot(t_ns, np.abs(sinal_tx), "--", color="#333333", label="Transmitido", zorder=3)
ax_linear.plot(t_ns, np.abs(sinal_rx), color=azul, label="Recebido")
ax_linear.set(ylabel="Magnitude |s(t)|", title="Envelope em escala linear")
ax_linear.set_ylim(bottom=0)

# Referencia comum: pico de potencia do pulso transmitido (nao o pico de RX).
referencia_tx = np.max(np.abs(sinal_tx)**2)
tx_db = 10 * np.log10(np.maximum(np.abs(sinal_tx)**2 / referencia_tx,
                                 10.0**(piso_db / 10)))
rx_db = 10 * np.log10(np.maximum(np.abs(sinal_rx)**2 / referencia_tx,
                                 10.0**(piso_db / 10)))
piso_potencia = 10.0**(piso_db / 10)
ax_db.plot(t_ns, np.maximum(np.abs(sinal_tx)**2 / referencia_tx, piso_potencia),
           "--", color="#333333", label="Transmitido", zorder=3)
ax_db.plot(t_ns, np.maximum(np.abs(sinal_rx)**2 / referencia_tx, piso_potencia),
           color=azul, label="Recebido")
ax_db.set_yscale("log")
ax_db.set(xlabel="Tempo relativo à primeira chegada (ns)",
           ylabel="Potência / pico TX (escala log)",
           title="Potência instantânea |s(t)|² — escala logarítmica",
           ylim=(piso_potencia, max(1.5, float(np.max(np.abs(sinal_rx)**2 / referencia_tx)) * 1.2)))

for ax in (ax_linear, ax_db):
    #ax.axvline(delta_t * 1e9, color=laranja, linestyle=":", label="Fim do pulso TX")
    ax.set_xlim(t_ns[0], t_ns[-1])
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right", fontsize=9)
grade_log(ax_db)

fim_ultima_copia = np.max(tau) + delta_t
#if fim_ultima_copia > t_plot[-1]:
#    ax_linear.text(
#        0.02, 0.96,
#        f"Janela incompleta: última cópia termina em {fim_ultima_copia * 1e9:.1f} ns",
#        transform=ax_linear.transAxes, va="top", fontsize=9, color="#9a3412",
#        bbox=dict(facecolor="#fff7ed", edgecolor="none", alpha=0.9)
#    )

pos_bs = np.array([x_tx, y_tx, h_tx], dtype=float)
pos_ut = np.array([x_rx, y_rx, h_rx], dtype=float)

velocidade = v_rx * np.asarray(v_rx_unit, dtype=float)

# Escala visual: comprimento da seta = velocidade × 10 segundos
tempo_seta = 10.0
seta = velocidade * tempo_seta
ponta_seta = pos_ut + seta

fig_geo = plt.figure(figsize=(9, 7))
ax_geo = fig_geo.add_subplot(111, projection="3d")

# Posições das antenas
ax_geo.scatter(
    *pos_bs, color="tab:blue", marker="^", s=100,
    label="Estação base (BS)"
)
ax_geo.scatter(
    *pos_ut, color="tab:orange", marker="o", s=70,
    label="Terminal (UT)"
)

# Alturas em relação ao solo
ax_geo.plot(
    [x_tx, x_tx], [y_tx, y_tx], [0, h_tx],
    color="tab:blue", linewidth=3
)
ax_geo.plot(
    [x_rx, x_rx],
    [y_rx, y_rx],
    [0, h_rx],
    color="tab:orange",
    linewidth=3
)

# Segmento geométrico BS–UT: não implica existência de LoS
ax_geo.plot(
    [x_tx, x_rx], [y_tx, y_rx], [h_tx, h_rx],
    "--", color="gray",
    label=f"Distância 3D: {d_txrx_3d:.2f} m"
)

# Projeção horizontal no plano z = 0
ax_geo.plot(
    [x_tx, x_rx], [y_tx, y_rx], [0, 0],
    ":", color="gray",
    label=f"Distância 2D: {d_txrx_2d:.2f} m"
)

# Vetor velocidade, representado por uma seta com escala explícita
if np.linalg.norm(velocidade) > 0:
    ax_geo.quiver(
        *pos_ut, *seta,
        color="tab:green",
        arrow_length_ratio=0.2,
        linewidth=2,
        label=f"Velocidade: {v_rx:.3f} m/s"
    )

ax_geo.text(x_tx, y_tx, h_tx + 2, "BS", color="tab:blue")
ax_geo.text(x_rx, y_rx, h_rx + 2, "UT", color="tab:orange")

# Limites incluindo solo, antenas e ponta da seta
pontos = np.vstack((
    pos_bs,
    pos_ut,
    ponta_seta,
    [x_tx, y_tx, 0],
    [x_rx, y_rx, 0]
))

margem = 0
lim_min = pontos.min(axis=0) - margem
lim_max = pontos.max(axis=0) + margem

ax_geo.set_xlim(lim_min[0], lim_max[0])
ax_geo.set_ylim(lim_min[1], lim_max[1])
ax_geo.set_zlim(lim_min[2], lim_max[2])

# Mesma escala espacial nos três eixos
ax_geo.set_box_aspect(lim_max - lim_min)

ax_geo.set_xlabel("X (m)")
ax_geo.set_ylabel("Y (m)")
ax_geo.set_zlabel("Altura Z (m)")
ax_geo.set_title(
    "Posições da base e do terminal\n"
    f"Seta verde: deslocamento equivalente a {tempo_seta:g} s"
)

ax_geo.view_init(elev=25, azim=-60)
#ax_geo.legend(loc="upper left", fontsize=9)
fig_geo.tight_layout()

plt.show()

