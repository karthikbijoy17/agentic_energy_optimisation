import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="BEOS",
    layout="wide"
)

st.markdown(
    """
    <style>

    .stApp {
        background-color: #050505;
    }

    h1, h2, h3 {
        color: white;
    }

    p, div {
        color: #E0E0E0;
    }

    [data-testid="stMetric"] {
        background-color: #111111;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #1F1F1F;
    }

    [data-testid="stDataFrame"] {
        border-radius: 12px;
    }

    .stAlert {
        background-color: #111111;
        border: 1px solid #222222;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    """
    <h1 style='margin-bottom:0px;'>
    🏢 BEOS
    </h1>

    <h3 style='color:#A0A0A0; margin-top:0px;'>
    Building Energy Optimization System
    </h3>

    <p style='color:#7A7A7A;'>
    Powered by RL + LLM
    </p>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "Dataset Analysis",
        "Custom Simulator",
        "Project Overview"
    ]
)

# ==================================================
# TAB 1 : DATASET ANALYSIS
# ==================================================

with tab1:

    st.header("Dataset Analysis")

    uploaded_file = st.file_uploader(
        "Upload Building Dataset",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        st.success("✅ Dataset Uploaded Successfully")

        # ------------------------------------------
        # Read Dataset
        # ------------------------------------------

        if uploaded_file.name.endswith(".csv"):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )

        # ------------------------------------------
        # Dataset Information
        # ------------------------------------------

        st.subheader("Dataset Information")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Rows",
                len(df)
            )

        with col2:

            st.metric(
                "Columns",
                len(df.columns)
            )

        # ------------------------------------------
        # Columns
        # ------------------------------------------

        st.subheader("Columns")

        st.write(
            list(df.columns)
        )

        # ------------------------------------------
        # Preview
        # ------------------------------------------

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head()
        )

        # ------------------------------------------
        # Missing Values
        # ------------------------------------------

        st.subheader("Missing Values Analysis")

        missing_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Missing Values": df.isnull().sum().values
            }
        )

        st.dataframe(
            missing_df
        )

        # ------------------------------------------
        # Power Consumption Distribution
        # ------------------------------------------

        if "Power Consumption" in df.columns:

            st.subheader(
                "Power Consumption Distribution"
            )

            fig, ax = plt.subplots()

            ax.hist(
                df["Power Consumption"],
                bins=30
            )

            ax.set_xlabel(
                "Power Consumption"
            )

            ax.set_ylabel(
                "Frequency"
            )

            st.pyplot(fig)

        # ------------------------------------------
        # Occupancy Distribution
        # ------------------------------------------

        if "Occupancy" in df.columns:

            st.subheader(
                "Occupancy Distribution"
            )

            occupancy_counts = (
                df["Occupancy"]
                .value_counts()
                .sort_index()
            )

            st.bar_chart(
                occupancy_counts
            )

        # ------------------------------------------
        # Temperature Distribution
        # ------------------------------------------

        if "Outdoor Temperature" in df.columns:

            temp_data = (
                df["Outdoor Temperature"]
                .dropna()
            )

            if len(temp_data) > 0:

                st.subheader(
                    "Temperature Distribution"
                )

                fig, ax = plt.subplots()

                ax.hist(
                    temp_data,
                    bins=30
                )

                ax.set_xlabel(
                    "Temperature"
                )

                ax.set_ylabel(
                    "Frequency"
                )

                st.pyplot(fig)

        # ==================================================
        # POLICY RESULTS ANALYSIS
        # ==================================================

        st.divider()

        st.header("Optimization Insights")

        try:

            policy_df = pd.read_csv(
                "data/policy_results.csv"
            )

            # ----------------------------
            # BEOS
            # ----------------------------

            avg_saving = policy_df[
                "Projected_Energy_Saving (%)"
            ].mean()

            avg_reward = policy_df[
                "Reward"
            ].mean()

            # Normalize Reward

            reward_score = min(
                100,
                max(
                    0,
                    (avg_reward + 1) * 50
                )
            )

            # Final BEOS

            beos = round(
                (
                    avg_saving * 0.7
                    +
                    reward_score * 0.3
                ),
                1
            )

            # ----------------------------
            # Grade
            # ----------------------------

            if beos >= 85:
                grade = "A+"

            elif beos >= 75:
                grade = "A"

            elif beos >= 65:
                grade = "B"

            elif beos >= 50:
                grade = "C"

            else:
                grade = "D"

            # ----------------------------
            # Opportunity
            # ----------------------------

            if avg_saving >= 30:
                opportunity = "HIGH"

            elif avg_saving >= 15:
                opportunity = "MEDIUM"

            else:
                opportunity = "LOW"

            # ----------------------------
            # Building Profile
            # ----------------------------

            top_action = (
                policy_df[
                    "Recommended_Action"
                ]
                .mode()[0]
            )

            profile_map = {

                "Reduce HVAC":
                    "HVAC Dominated",

                "Reduce Lighting":
                    "Lighting Dominated",

                "Check Equipment":
                    "Equipment Intensive",

                "Do Nothing":
                    "Already Efficient"
            }

            building_profile = profile_map.get(
                top_action,
                "Balanced"
            )

            # ----------------------------
            # KPI CARDS
            # ----------------------------

            st.subheader(
                "Building Intelligence"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "BEOS",
                    f"{beos}/100"
                )

            with c2:

                st.metric(
                    "Building Grade",
                    grade
                )

            with c3:

                st.metric(
                    "Optimization Opportunity",
                    f"{avg_saving:.1f}%"
                )

            with c4:

                st.metric(
                    "Building Profile",
                    building_profile
                )

            st.divider()

            st.subheader(
                "Executive Summary"
            )

            st.info(
                f"""
                The building shows a BEOS score of {beos}/100 with a
                {grade} performance rating.

                The RL agent identified an average optimization
                opportunity of {avg_saving:.1f}%.

                The dominant operational pattern is
                '{building_profile}'.

                Most optimization recommendations focus on
                '{top_action}'.
                """
            )

            st.subheader(
                "Recommended Action Distribution"
            )

            action_counts = (
                policy_df[
                    "Recommended_Action"
                ]
                .value_counts()
            )

            st.bar_chart(
                action_counts
            )

            

          

            st.divider()

            st.subheader(
                "Building Personality"
            )

            if top_action == "Reduce HVAC":

                personality = """
                HVAC-Dominated Building

                Most optimization opportunities are related to HVAC operations,
                indicating cooling/heating systems are the primary energy consumers.
                """

            elif top_action == "Reduce Lighting":

                personality = """
                Lighting-Dominated Building

                Lighting systems contribute significantly to energy consumption,
                creating optimization opportunities.
                """

            elif top_action == "Check Equipment":

                personality = """
                Equipment-Intensive Building

                Equipment and machinery appear to be major contributors
                to energy usage.
                """

            else:

                personality = """
                Balanced Energy Consumer

                No major inefficiencies were detected.
                Energy consumption appears relatively balanced.
                """

            st.success(personality)

            st.subheader(
                "Top Energy Waste Patterns"
            )

            waste_states = (
                policy_df
                .sort_values(
                    by="Projected_Energy_Saving (%)",
                    ascending=False
                )
                .head(5)
            )

            for idx, row in waste_states.iterrows():

                st.write(
                    f"""
                    • State: {row['Current_State']}

                    • Recommended Action:
                    {row['Recommended_Action']}

                    • Saving Potential:
                    {row['Projected_Energy_Saving (%)']}%
                    """
                )

            st.divider()

            # ----------------------------
            # RESEARCH FINDINGS
            # BUG FIX: moved inside try block
            # ----------------------------

            st.subheader(
                "Research Findings"
            )

            st.markdown(
                f"""
                • {len(policy_df)} unique operational states were analyzed.

                • The reinforcement learning agent identified an average
                optimization opportunity of {avg_saving:.2f}%.

                • The dominant optimization strategy was
                **{top_action}**.

                • The building exhibits a
                **{building_profile}** energy behavior pattern.

                • The overall BEOS score is
                **{beos}/100**.
                """
            )

        except Exception as e:

            st.error(
                f"Could not load policy_results.csv: {e}"
            )

# ==================================================
# TAB 2 : CUSTOM SIMULATOR
# ==================================================

with tab2:

    st.header(
        "Energy Optimization Simulator"
    )

    st.markdown(
        """
        Simulate a building condition and
        see how the LLM optimization agent
        would respond.
        """
    )

    st.divider()

    col1, col2 = st.columns(2)

    # -------------------------
    # INPUTS
    # -------------------------

    with col1:

        power = st.slider(
            "Power Level",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1
        )

        occupancy_level = st.select_slider(
            "Occupancy Level",
            options=[
                "Empty",
                "Low",
                "Medium",
                "High"
            ],
            value="Medium"
        )

        # Convert to internal value

        if occupancy_level == "Empty":
            occupancy = 0.0

        elif occupancy_level == "Low":
            occupancy = 0.3

        elif occupancy_level == "Medium":
            occupancy = 0.6

        else:
            occupancy = 1.0

        temperature = st.slider(
            "Temperature Level",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )

    # -------------------------
    # BUILD STATE
    # -------------------------

    state = (
        round(power, 1),
        occupancy_level,
        round(temperature, 1)
    )

    # -------------------------
    # ACTION RULES
    # -------------------------

    if power >= 0.7:

        action = "Reduce HVAC"

        saving = 30

    elif power >= 0.5:

        action = "Reduce Lighting"

        saving = 15

    elif power >= 0.3:

        action = "Check Equipment"

        saving = 8

    else:

        action = "Do Nothing"

        saving = 0

    # -------------------------
    # OUTPUTS
    # -------------------------

    with col2:

        st.subheader(
            "Agent Recommendation"
        )

        st.metric(
            "Recommended Action",
            action
        )

        st.metric(
            "Projected Saving",
            f"{saving}%"
        )

        st.metric(
            "Current State",
            str(state)
        )

    st.divider()

    # -------------------------
    # COUNTERFACTUAL
    # -------------------------

    st.subheader(
        "Counterfactual Analysis"
    )

    if action == "Reduce HVAC":

        alternative_action = "Do Nothing"
        alternative_saving = 0

    elif action == "Reduce Lighting":

        alternative_action = "Do Nothing"
        alternative_saving = 0

    elif action == "Check Equipment":

        alternative_action = "Do Nothing"
        alternative_saving = 0

    else:

        alternative_action = "No Alternative"
        alternative_saving = 0

    colA, colB = st.columns(2)

    with colA:

        st.success(
            f"""
            Agent Decision

            Action:
            {action}

            Expected Saving:
            {saving}%
            """
        )

    with colB:

        st.warning(
            f"""
            Alternative Scenario

            Action:
            {alternative_action}

            Expected Saving:
            {alternative_saving}%
            """
        )

    saving_difference = (
        saving
        -
        alternative_saving
    )

    st.metric(
        "Optimization Advantage",
        f"{saving_difference}%"
    )

    st.info(
        f"""
        The Agent selected '{action}' because it provides
        approximately {saving_difference}% higher
        energy optimization potential than the
        baseline operating condition.
        """
    )

    st.divider()

    # -------------------------
    # AGENT DECISION AUDITOR
    # -------------------------

    st.subheader(
        "Agent Decision Auditor"
    )

    if action == "Reduce HVAC":

        observation = (
            "The building is consuming a high amount "
            "of energy through HVAC operations."
        )

        reasoning = (
            "Power demand is elevated and cooling/heating "
            "systems appear to be the dominant load."
        )

        recommendation = (
            "Reduce HVAC intensity or adjust setpoints "
            "to improve energy efficiency."
        )

    elif action == "Reduce Lighting":

        observation = (
            "Lighting systems appear to be consuming "
            "more energy than expected."
        )

        reasoning = (
            "Current power levels indicate potential "
            "lighting inefficiencies."
        )

        recommendation = (
            "Reduce lighting usage in low-priority "
            "areas when possible."
        )

    elif action == "Check Equipment":

        observation = (
            "Equipment energy usage may be contributing "
            "to avoidable consumption."
        )

        reasoning = (
            "The current state suggests that equipment "
            "performance should be reviewed."
        )

        recommendation = (
            "Inspect equipment operation and maintenance "
            "conditions."
        )

    else:

        observation = (
            "The building appears to be operating "
            "efficiently."
        )

        reasoning = (
            "No major inefficiencies were detected "
            "in the current state."
        )

        recommendation = (
            "Continue current operating strategy."
        )

    st.markdown(
        f"""
### Observation
{observation}

### Reasoning
{reasoning}

### Recommendation
{recommendation}
        """
    )

    st.divider()

    # -------------------------
    # BUILDING HEALTH INDICATOR
    # -------------------------

    st.subheader(
        "Building Health Indicator"
    )

    health_score = 100 - saving

    if health_score >= 80:

        health_status = "Excellent"

    elif health_score >= 60:

        health_status = "Good"

    elif health_score >= 40:

        health_status = "Moderate"

    else:

        health_status = "Critical"

    st.metric(
        "Building Health Score",
        f"{health_score}/100"
    )

    st.progress(
        health_score / 100
    )

    st.write(
        f"Current Status: **{health_status}**"
    )

# ==================================================
# TAB 3 : PROJECT OVERVIEW
# ==================================================

with tab3:

    st.header("Project Overview")

    st.markdown(
        """
        ### Building Energy Optimization System

        This project combines:

        - Reinforcement Learning (Q-Learning)
        - Energy Optimization
        - Decision Intelligence
        - LLM-Based Explainability

        ### Workflow

        Dataset  
        → State Representation  
        → Q-Learning Agent  
        → Decision Agent  
        → LLM Decision Auditor  
        → Dashboard
        """
    )

    st.divider()

    st.subheader(
        "Mathematical Foundations"
    )

    st.markdown(
        r"""
### 1. State Representation

The building state is represented as:

\[
State = (Power,\ Occupancy,\ Temperature)
\]

Example:

\[
(0.8,\ 1,\ 0.6)
\]

where:

- Power = normalized energy consumption
- Occupancy = occupancy condition
- Temperature = normalized temperature level

---

### 2. Energy Saving

The energy saving achieved after an action is:

\[
EnergySaving =
Power_{before}
-
Power_{after}
\]

This measures the reduction in energy consumption.

---

### 3. Reward Function

The reinforcement learning agent learns using:

\[
Reward =
EnergySaving
-
ComfortPenalty
\]

where:

\[
ComfortPenalty =
Occupancy
\times
TemperatureIncrease
\]

This balances energy efficiency and occupant comfort.

---

### 4. Confidence Estimation

The confidence of a recommendation is calculated as:

\[
Confidence =
\frac{BestQ}
{\sum |Q|}
\times 100
\]

where:

- BestQ = highest Q-value for the state
- Σ|Q| = sum of absolute Q-values

Higher confidence indicates a stronger preference for the selected action.

---

### 5. Q-Learning Update Rule

The policy is learned using:

\[
Q(s,a)
=
Q(s,a)
+
\alpha
\Big[
R
+
\gamma
\max Q(s',a')
-
Q(s,a)
\Big]
\]

where:

- α = learning rate
- γ = discount factor
- R = reward
- Q(s,a) = action value

This allows the agent to continuously improve its decision-making policy.
"""
    )