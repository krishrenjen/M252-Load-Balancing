import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CORE PHYSICS ENGINE ---
def get_derivatives(model_type, state, params, proc_mode):
    n = len(state)
    dxdt = np.zeros(n)
    eps = 1e-6
    
    k = params['k']
    mu = params['mu']
    p_max = params['p_max']
    
    # --- Step A: Calculate Arrival Rates (Lambda_i) ---
    lams = np.zeros(n)
    
    # Algorithmic models use Global Lambda (Λ)
    if model_type in ["Round Robin", "Weighted Round Robin", "Least Connections", "Weighted Least Connections"]:
        lam_total = params['lam_total']
        
        # In Saturated mode, we use p_max for weights; in Proportional, we use mu
        weights = p_max if (proc_mode == "Saturated" or model_type == "Basic Model (Constant Rate)") else mu
        
        if model_type == "Round Robin":
            # λ_i = Λ / n 
            lams = np.full(n, lam_total / n)
            
        elif model_type == "Weighted Round Robin":
            # λ_i distributed by processing power 
            lams = lam_total * (weights / (np.sum(weights) + eps))
            
        elif model_type == "Least Connections":
            # λ_i distributed to those with lower load x_i
            total_x = np.sum(state) + eps
            lams = lam_total * ((total_x - state) / ((n-1) * total_x + eps))
            
        elif model_type == "Weighted Least Connections":
            # λ_i distributed by expected wait time W_i = x_i / capacity
            w = state / (weights + eps)
            total_w = np.sum(w) + eps
            lams = lam_total * ((total_w - w) / ((n-1) * total_w + eps))
    else: 
        # Decoupled models use individual static arrival rates (λ_i)
        lams = params['lams_fixed']

    # --- Step B: Calculate Processing and Coupling ---
    for i in range(n):
        # 1. Basic Model: Uses Constant Rate p_i 
        if model_type == "Basic Model (Constant Rate)":
            proc = p_max[i] if state[i] > 0 else 0
        
        # 2. Advanced Models: Toggle between Proportional and Saturated
        else:
            if proc_mode == "Saturated":
                # S(x) = x / (x + 1) 
                proc = p_max[i] * (state[i] / (state[i] + 1))
            else:
                # Proportional logic: mu * x 
                proc = mu[i] * state[i]
            
        # Coupling (Work Stealing k): k * sum(x_j - x_i) 
        coupling = 0
        for j in range(n):
            if i != j:
                coupling += k * (state[j] - state[i])
        
        dxdt[i] = lams[i] - proc + coupling
        
    return dxdt

# --- 2. NUMERICAL INTEGRATOR (RK4) ---
def solve_system(model_type, params, proc_mode, t_max, dt, init_val):
    n_steps = int(t_max / dt)
    n_servers = len(init_val)
    t = np.linspace(0, t_max, n_steps)
    x = np.zeros((n_steps, n_servers))
    x[0] = init_val
    
    for i in range(0, n_steps - 1):
        k1 = dt * get_derivatives(model_type, x[i], params, proc_mode)
        k2 = dt * get_derivatives(model_type, x[i] + k1/2, params, proc_mode)
        k3 = dt * get_derivatives(model_type, x[i] + k2/2, params, proc_mode)
        k4 = dt * get_derivatives(model_type, x[i] + k3, params, proc_mode)
        # Apply max(0, x) constraint
        x[i+1] = np.maximum(0, x[i] + (k1 + 2*k2 + 2*k3 + k4) / 6)
        
    return t, x

# --- 3. STREAMLIT INTERFACE ---
st.set_page_config(page_title="Server Load Balancer Lab", layout="wide")
st.title("Server Load Balancing Simulation")

with st.sidebar:
    st.header("1. Model Selection")
    model_choice = st.selectbox("Arrival Algorithm", 
                                ["Basic Model (Constant Rate)", 
                                 "Decoupled Static", 
                                 "Round Robin", 
                                 "Weighted Round Robin", 
                                 "Least Connections", 
                                 "Weighted Least Connections"])
    
    is_algorithmic = "Round Robin" in model_choice or "Connections" in model_choice
    
    # Handle the mode and UI visibility
    if model_choice == "Basic Model (Constant Rate)":
        proc_mode = "Saturated" # Forced logic
        st.info("Basic Model uses constant rates ($p_i$).")
    else:
        proc_mode = st.radio("Processing Mode", ["Proportional", "Saturated"], index=0)

    n_servers = st.radio("Number of Servers", [2, 3])

    st.header("2. Global Parameters")
    if is_algorithmic:
        lam_total = st.slider("Total Arrival $(\Lambda)$", 0.0, 15.0, 5.0)
    else:
        lam_total = 0.0

    k_gain = st.slider("Coupling Gain $(k)$", 0.0, 2.0, 0.2)
    
    st.header("3. Server Metrics")
    mu_vals = np.zeros(n_servers)
    p_vals = np.zeros(n_servers)
    lams_fixed = np.zeros(n_servers)
    
    for i in range(n_servers):
        with st.expander(f"Server {i+1} Parameters"):
            # Hide mu if in Saturated mode or Basic Model
            if proc_mode == "Proportional" and model_choice != "Basic Model (Constant Rate)":
                mu_vals[i] = st.slider(f"Processing Coeff $(\mu_{i+1})$", 0.05, 2.0, 0.5)
            else:
                mu_vals[i] = 1.0 # Default fallback
            
            # Hide p_max if in Proportional mode
            if proc_mode == "Saturated" or model_choice == "Basic Model (Constant Rate)":
                p_vals[i] = st.slider(f"Max Processing Rate $(p_{i+1})$", 0.1, 10.0, 3.0)
            else:
                p_vals[i] = 1.0 # Default fallback
            
            if not is_algorithmic:
                lams_fixed[i] = st.slider(f"Arrival Rate $(\lambda_{i+1})$", 0.0, 10.0, 2.0)

    t_limit = st.number_input("Simulation Duration", value=100)

# Run Simulation
params = {'k': k_gain, 'mu': mu_vals, 'p_max': p_vals, 'lam_total': lam_total, 'lams_fixed': lams_fixed}
t_series, data = solve_system(model_choice, params, proc_mode, t_limit, 0.01, np.ones(n_servers) * 2.0)

# --- 4. PLOTTING ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Load over Time")
    fig_ts, ax_ts = plt.subplots(figsize=(10, 5))
    for i in range(n_servers):
        ax_ts.plot(t_series, data[:, i], label=f"Server {i+1}", lw=2)
    ax_ts.set_ylabel("Load $x(t)$")
    ax_ts.set_xlabel("Time")
    ax_ts.grid(True, alpha=0.3)
    ax_ts.legend()
    st.pyplot(fig_ts)

with col2:
    if n_servers == 2:
        st.subheader("Phase Portrait")
        fig_ph, ax_ph = plt.subplots()
        ax_ph.plot(data[:, 0], data[:, 1], color='purple', alpha=0.7)
        
        x_m = max(np.max(data)*1.1, 1.0)
        u, v = np.meshgrid(np.linspace(0, x_m, 12), np.linspace(0, x_m, 12))
        u_d, v_d = np.zeros(u.shape), np.zeros(v.shape)
        for i in range(12):
            for j in range(12):
                d = get_derivatives(model_choice, np.array([u[i,j], v[i,j]]), params, proc_mode)
                u_d[i,j], v_d[i,j] = d[0], d[1]
        ax_ph.quiver(u, v, u_d, v_d, color='gray', alpha=0.3)
        ax_ph.set_xlabel("$x_1$")
        ax_ph.set_ylabel("$x_2$")
        st.pyplot(fig_ph)
    else:
        st.info("Phase portraits are available for 2-server models.")