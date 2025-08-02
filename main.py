import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import imageio
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

MATERIAL_DB = {
    "iron": {"J_per_kB": 21.1, "Tc_exp": 1043},
    "k2cof4": {"J_per_kB": 10.0, "Tc_exp": 110},
    "rb2cof4": {"J_per_kB": 7.0, "Tc_exp": 100},
    "dypo4": {"J_per_kB": 2.5, "Tc_exp": 3.4}
}

st.title(" 2D Ising Model: Magnetic Phase Transition, Spin States, and Animation")
#st.markdown("""<h1 style="text-align: center;">2D Ising Model: Magnetic Phase Transition, Spin States, and Animation; font-family: monospace:'></h1>""")
st.markdown("""
---
### **Usage: Step-by-Step Guide**

1. **Sidebar se parameters set karein** (material, lattice size, MC steps, random seed, etc.).
2. **"Run Simulation" button dabayein.**
3. **Tabs ka use karke:**
   - Graphs, phase diagrams, spin states, aur animations analyze karein.
4. **Her tab ke neeche likhe explanation padhein — phase transition samjho!**
5. **Chahe toh apna data bhi compare karein (experimental upload).**

---
""")

material = st.sidebar.selectbox(
    "Choose Real Material:",
    options=list(MATERIAL_DB.keys()),
    format_func=lambda x: x.upper()
)
params = MATERIAL_DB[material]
JkB, Tc_exp = params["J_per_kB"], params["Tc_exp"]
st.sidebar.info(f"**{material.upper()}**: J/kB = {JkB} K, Tc(exp) = {Tc_exp} K")

N        = st.sidebar.slider("Lattice Size (N×N)", 10, 64, 30)
n_eq     = st.sidebar.number_input("Equilibration Steps", 500, step=100)
n_samples= st.sidebar.number_input("Samples per T", 200, step=100)
seed     = st.sidebar.number_input("Random Seed", 0, step=1)
minT     = st.sidebar.number_input("Low Temperature (K)", int(Tc_exp * 0.7))
maxT     = st.sidebar.number_input("High Temperature (K)", int(Tc_exp * 1.3))
nT       = st.sidebar.slider("Number of Temperatures", 10, 50, 30)
run_sim  = st.sidebar.button("Run Simulation")

# Animation controls in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Animation Controls:**")
spin_anim_T = st.sidebar.number_input("Animation: Temperature (K)", min_value=int(minT), max_value=int(maxT), value=int(Tc_exp))
spin_anim_steps = st.sidebar.slider("Spin Animation MC Steps", 30, 200, 80)
temp_anim_steps = st.sidebar.slider("Temp Sweep MC Steps", 100, 800, 400)

exp_data = None
uploaded = st.sidebar.file_uploader("Upload experimental CSV (T[K],M[])", type=['csv'])
if uploaded:
    exp_data = pd.read_csv(uploaded)
    st.sidebar.success("Experimental file loaded.")

def initial_lattice(N, key):
    return 2 * jax.random.randint(key, (N, N), 0, 2) - 1

@jax.jit
def checkerboard_update(spins, beta, key):
    N = spins.shape[0]
    for offset in [0, 1]:
        mask = jnp.fromfunction(lambda i, j: ((i + j) % 2 == offset), (N, N), dtype=jnp.int32).astype(bool)

        neighbors = (jnp.roll(spins, 1, axis=0) + jnp.roll(spins, -1, axis=0) +
                     jnp.roll(spins, 1, axis=1) + jnp.roll(spins, -1, axis=1))
        key, subkey = jax.random.split(key)
        rand_mat = jax.random.uniform(subkey, (N, N))
        deltaE = 2 * spins * neighbors
        flip = (deltaE < 0) | (rand_mat < jnp.exp(-beta * deltaE))
        spins = jnp.where(mask & flip, -spins, spins)
    return spins

def calc_energy(state):
    E = -jnp.sum(state * jnp.roll(state, 1, 0)) - jnp.sum(state * jnp.roll(state, 1, 1))
    return E / 2.0

def calc_magnetization(state):
    return jnp.sum(state)

def ising_sim(N, n_eq, n_samples, T_arr, JkB, seed, Tc_exp, maxT):
    E_av, M_av, C_av, X_av = [], [], [], []
    spins_below_tc, spins_above_tc = None, None
    key = jax.random.PRNGKey(seed)
    for T_real in T_arr:
        T_code = T_real / JkB
        beta = 1.0 / T_code
        skey, key = jax.random.split(key)
        state = initial_lattice(N, skey)
        for _ in range(n_eq):
            skey, key = jax.random.split(key)
            state = checkerboard_update(state, beta, skey)
        E_samples = []
        M_samples = []
        for _ in range(n_samples):
            skey, key = jax.random.split(key)
            state = checkerboard_update(state, beta, skey)
            E_samples.append(calc_energy(state))
            M_samples.append(jnp.abs(calc_magnetization(state)))
        E_samples = jnp.array(E_samples)
        M_samples = jnp.array(M_samples)
        E_mean = float(jnp.mean(E_samples) / (N*N))
        M_mean = float(jnp.mean(M_samples) / (N*N))
        C_mean = float(jnp.var(E_samples)/(T_code**2 * N**2))
        X_mean = float(jnp.var(M_samples)/(T_code * N**2))
        E_av.append(E_mean)
        M_av.append(M_mean)
        C_av.append(C_mean)
        X_av.append(X_mean)
        if spins_below_tc is None and T_real < Tc_exp and T_real > Tc_exp - 0.25*(maxT-minT):
            spins_below_tc = np.array(state)
        if spins_above_tc is None and T_real > Tc_exp and T_real < Tc_exp + 0.25*(maxT-minT):
            spins_above_tc = np.array(state)
    return np.array(E_av), np.array(M_av), np.array(C_av), np.array(X_av), spins_below_tc, spins_above_tc

def animate_spin_evolution(N, MC_steps, beta, seed, filename="spin_evolution.gif"):
    key = jax.random.PRNGKey(seed+1337)
    state = initial_lattice(N, key)
    frames = []
    for k in range(MC_steps):
        key, subkey = jax.random.split(key)
        state = checkerboard_update(state, beta, subkey)
        frames.append(np.array(state))
    images = []
    for i, s in enumerate(frames):
        fig, ax = plt.subplots()
        ax.imshow(s, cmap="bwr", vmin=-1, vmax=1)
        ax.set_title(f"MC Step {i+1}")
        ax.axis('off')
        fig.tight_layout()
        fig.canvas.draw()  # Ensure the renderer exists
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,))

        images.append(img)
        plt.close(fig)
    imageio.mimsave(filename, images, duration=0.1)
    return filename

def animate_temp_sweep(N, T_arr, MC_steps, seed, JkB, filename="ising_temp_sweep.gif"):
    key = jax.random.PRNGKey(seed+4242)
    images = []
    for i, T_real in enumerate(T_arr):
        T_code = T_real / JkB
        beta = 1.0 / T_code
        key, subkey = jax.random.split(key)
        state = initial_lattice(N, subkey)
        for _ in range(MC_steps):
            key, subkey = jax.random.split(key)
            state = checkerboard_update(state, beta, subkey)
        fig, ax = plt.subplots()
        ax.imshow(np.array(state), cmap="bwr", vmin=-1, vmax=1)
        ax.axis('off')
        ax.set_title(f"T = {T_real:.2f}")
        fig.tight_layout()
        fig.canvas.draw()
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(canvas.get_width_height()[::-1] + (4,))

        images.append(img)
        plt.close(fig)
    imageio.mimsave(filename, images, duration=0.12)
    return filename

# --- MAIN RUN ---
if run_sim:
    T_real_arr = np.linspace(minT, maxT, nT)
    with st.spinner("Running Ising simulation (JAX-GPU/CPU)..."):
        E, M, C, X, spins_below_tc, spins_above_tc = ising_sim(
            N, n_eq, n_samples, T_real_arr, JkB, seed, Tc_exp, maxT
        )

    tabs = st.tabs(["Magnetization vs Temperature", "Heat Capacity vs Temperature", "Phase Diagram", "Animations"])

    with tabs[0]:
        st.subheader("Magnetization vs Temperature")
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(T_real_arr, M, 'o-', label="Simulation")
        if exp_data is not None:
            ax.plot(exp_data.iloc[:,0], exp_data.iloc[:,1], 's--', label="Experiment", color='orange')
        ax.axvline(Tc_exp, color='red', ls=':', label=f"Tc (Exp)={Tc_exp}K", lw=2)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Magnetization per spin")
        ax.legend()
        ax.grid()
        st.pyplot(fig)
        st.markdown("""
        **Analysis:** 
        - **$T < T_c$:** System ferromagnetic—magnetization high, spins mostly aligned. 
        - **$T = T_c$ (vertical red line):** Magnetization sharply falls—this is the phase transition.
        - **$T > T_c$:** Magnetization ~0—system is paramagnetic; spins are random.
        """)

    with tabs[1]:
        st.subheader("Heat Capacity vs Temperature")
        fig2, ax2 = plt.subplots(figsize=(7,4))
        ax2.plot(T_real_arr, C, 'd-', label="Heat Capacity (Sim)")
        ax2.axvline(Tc_exp, color='red', ls=':', label=f"Transition $T_c$={Tc_exp}K", lw=2)
        ax2.set_xlabel("Temperature (K)")
        ax2.set_ylabel("Specific Heat (per site)")
        ax2.legend()
        ax2.grid()
        st.pyplot(fig2)
        st.markdown("""
        **Explanation:** 
        - $T_c$ (red line) par heat capacity mein sharp peak—ye transition ki jagah hain. 
        - $T < T_c$ mein heat capacity moderate, $T > T_c$ mein phir se kam ho jata hai.
        """)

    with tabs[2]:
        st.subheader("2D Ising Model Phase Diagram (with Spin States)")
        fig3, ax3 = plt.subplots(figsize=(7,4))
        ax3.axvspan(minT, Tc_exp, alpha=0.3, color='blue', label="Ferromagnetic")
        ax3.axvspan(Tc_exp, maxT, alpha=0.3, color='orange', label="Paramagnetic")
        ax3.axvline(Tc_exp, color='red', ls='--', label="Transition $T_c$")
        ax3.plot(T_real_arr, M, 'ko', markersize=3, label="Magnetization")
        ax3.set_xlabel("Temperature (K)")
        ax3.set_ylabel("Phase")
        ax3.set_yticks([])
        ax3.legend()
        st.pyplot(fig3)
        st.markdown(f"""
        **Phase Diagram Analysis:** 
        - **Blue area $T < T_c$:** Ferromagnetic phase (spins aligned, high magnetization).
        - **Red line at $T_c$:** Transition—magnetization drops.
        - **Orange area $T > T_c$:** Paramagnetic phase (spins random, $m \sim 0$).
        - **Now, let's see the spin configuration snapshots below!**
        """)
        if spins_below_tc is not None and spins_above_tc is not None:
            fig4, axes = plt.subplots(1,2, figsize=(8,3))
            axes[0].imshow(spins_below_tc, cmap='bwr', vmin=-1, vmax=1)
            axes[0].set_title(r"Below $T_c$:\nFerromagnetic (Ordered)")
            axes[0].axis('off')
            axes[1].imshow(spins_above_tc, cmap='bwr', vmin=-1, vmax=1)
            axes[1].set_title(r"Above $T_c$:\nParamagnetic (Random)")
            axes[1].axis('off')
            plt.tight_layout()
            st.pyplot(fig4)
            st.markdown("""
            **Interpretation:** 
            - *Left (Below $T_c$):* Spins mostly same color—system magnetized, ordered.
            - *Right (Above $T_c$):* Red/blue mixed—spins randomly oriented, system unmagnetized.
            """)
        else:
            st.warning("Spin snapshots not available for current simulation range/settings.")

    # --- Animations Tab ---
    with tabs[3]:
        st.subheader("Animations: Ising Model Dynamics")

        # 1. Lattice evolution at a single T (user can pick T)
        st.markdown(f"""
        **Spin Evolution at Fixed Temperature** 
        Temperature: **{spin_anim_T} K** | Steps: **{spin_anim_steps}**
        """)
        beta_anim = 1.0 / (spin_anim_T / JkB)
        gif1 = "spin_evolution.gif"
        if (not os.path.exists(gif1)) or st.button("Generate Spin Evolution Animation"):
            with st.spinner("Building spin evolution animation..."):
                animate_spin_evolution(N, spin_anim_steps, beta_anim, seed, gif1)
        st.image(gif1, caption="Time evolution of spin lattice")

        # 2. Lattice snapshots as you sweep T from minT to maxT
        st.markdown(f"""
        **Spin Snapshots Across Temperature Sweep** 
        MC Steps per T: **{temp_anim_steps}**
        """)
        gif2 = "ising_temp_sweep.gif"
        T_anim_arr = np.linspace(minT, maxT, min(28, nT))  # Fewer frames for speed
        if (not os.path.exists(gif2)) or st.button("Generate Temperature Sweep Animation"):
            with st.spinner("Building temperature sweep animation..."):
                animate_temp_sweep(N, T_anim_arr, temp_anim_steps, seed, JkB, gif2)
        st.image(gif2, caption="Spin patterns as T increases (phase boundary becomes visible)")

        st.markdown("""
        - **Top animation:** At a fixed temperature, see thermalization and spin ordering/disordering over time. 
        - **Bottom animation:** As temperature increases, see the system go from ordered (magnetized) to random (paramagnetic)—the phase transition in action.
        """)

    st.success(
        f"""
**FERROMAGNETIC:** $T < {Tc_exp}$ K 
**PHASE TRANSITION at $T_c={Tc_exp}$ K:** Magnetization goes from non-zero to zero 
**PARAMAGNETIC:** $T > {Tc_exp}$ K 
        """
    )
    if exp_data is not None:
        try:
            interp_sim = np.interp(exp_data.iloc[:,0], T_real_arr, M)
            rmse = np.sqrt(np.mean((exp_data.iloc[:,1] - interp_sim) ** 2))
            st.info(f"Simulation vs Experiment RMSE: {rmse:.4f}")
        except Exception:
            pass

    st.markdown("""
    ---
    **Key Physics:** 
    - *Ferromagnetism*: Strong spin order, spontaneous magnetization.
    - *Phase transition ($T_c$)*: Order lost, magnetization drops suddenly, system becomes random.
    - *Paramagnetism*: No net alignment—thermal energy overcomes ordering.
    - *Diamagnetism*: 2D Ising model me nahi hota—sirf ferro/para dikhta hai.

    **Animations show:**
    - *How the spin lattice evolves at one T (thermalization or fluctuations).*
    - *How long-range order collapses as T approaches and exceeds $T_c$ (the heart of a phase transition).*
    """)