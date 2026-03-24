import streamlit as st

st.set_page_config(
    page_title="QWAN Complexity Lab",
    layout="wide"
)

st.title("QWAN Complexity Lab")

st.sidebar.title("Modules")

module = st.sidebar.radio(
    "Select Module",
    [
        "Overview",
        "Chess Engine",
        "Cardiology Simulation"
    ]
)

if module == "Overview":

    st.markdown("""
    ## QWAN Research Platform

    Integrated framework for:

    - Complex systems monitoring
    - Econophysics diagnostics
    - Neural search systems
    - Experimental chess AI
    """)

elif module == "Chess Engine":

    import chess_page

elif module == "Cardiology Simulation":

    from metastablex.dynamics.cardiology import simulate_hrv

    import numpy as np
    import matplotlib.pyplot as plt

    st.subheader("Heart Rate Variability Simulation")

    regime = st.selectbox(
        "Regime",
        ["healthy", "rigid", "chaotic"]
    )

    n = st.slider("Length", 500, 3000, 1500)

    if st.button("Run Simulation"):

        rr = simulate_hrv(n=n, regime=regime)

        hr = 60 / rr

        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(hr)

        st.pyplot(fig)

elif module == "Chess Engine":

    import chess_page
