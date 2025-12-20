import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import tensorflow as tf
import plotly.express as px
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

# ==============================
# Streamlit Config
# ==============================
st.set_page_config(
    page_title="PrognosAI - RUL Prediction",
    layout="wide"
)

# ==============================
# Custom Attention Layer
# ==============================
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = inputs * alpha
        return K.sum(context, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# ==============================
# Load Model & Scaler
# ==============================
@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"AttentionLayer": AttentionLayer}
    )
    scaler = joblib.load(scaler_path)
    return model, scaler

# ==============================
# Load Sequences (from disk)
# ==============================
def load_sequences(seq_path):
    return np.load(seq_path)

# ==============================
# Visualization Functions
# ==============================
def plot_rul_trends(df):
    st.subheader("📈 RUL Degradation Trends")

    engines = sorted(df["engine_id"].unique())
    selected = st.multiselect(
        "Select up to 5 Engines",
        engines,
        default=engines[:5],
        max_selections=5
    )

    df_plot = df[df["engine_id"].isin(selected)] if selected else df

    fig = px.line(
        df_plot,
        x="cycle",
        y="RUL",
        color="engine_id",
        labels={"cycle": "Cycle", "RUL": "Predicted RUL"},
        title="Remaining Useful Life Over Cycles"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_alert_zone_counts(df):
    latest = df.groupby("engine_id").tail(1).reset_index(drop=True)

    bins = [-1, 10, 30, float("inf")]
    labels = ["Critical", "Warning", "Safe"]
    latest["Alert"] = pd.cut(latest["RUL"], bins=bins, labels=labels)

    counts = latest["Alert"].value_counts().reindex(labels).fillna(0).reset_index()
    counts.columns = ["Alert Zone", "Engines"]

    fig = px.bar(
        counts,
        x="Alert Zone",
        y="Engines",
        color="Alert Zone",
        text="Engines",
        title="🚨 Engine Health Distribution",
        color_discrete_map={
            "Critical": "#dc3545",
            "Warning": "#ffc107",
            "Safe": "#28a745"
        }
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

def render_dashboard(df):
    st.subheader("📊 Maintenance Status Overview")

    latest = df.groupby("engine_id").tail(1).reset_index(drop=True)
    latest["Alert"] = pd.cut(
        latest["RUL"],
        bins=[-1, 10, 30, float("inf")],
        labels=["Critical", "Warning", "Safe"]
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Critical", (latest["Alert"] == "Critical").sum())
    c2.metric("🟡 Warning", (latest["Alert"] == "Warning").sum())
    c3.metric("🟢 Safe", (latest["Alert"] == "Safe").sum())

    st.dataframe(
        latest[["engine_id", "cycle", "RUL", "Alert"]].round(2),
        use_container_width=True
    )

# ==============================
# Main App
# ==============================
def main():
    st.title("🛠️ PrognosAI – RUL Prediction Dashboard")

    # Fixed paths
    model_path = "models/bidirectional_attention_model.keras"
    scaler_path = "data/processed/train/scaler.pkl"
    seq_path = "data/processed/test/rolling_window_sequences.npy"

    # Check required server files
    for p in [model_path, scaler_path, seq_path]:
        if not os.path.exists(p):
            st.error(f"❌ Missing file: {p}")
            return

    # Load model
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    st.success("✅ BiLSTM + Attention model loaded")

    # Load sequences
    X_test = load_sequences(seq_path)
    st.success(f"✅ Sequences loaded: {X_test.shape}")

    # Metadata upload
    uploaded_meta = st.file_uploader(
        "📋 Upload Metadata CSV",
        type=["csv"]
    )

    if uploaded_meta is None:
        st.info("👆 Please upload metadata.csv to continue")
        return

    df_meta = pd.read_csv(uploaded_meta)

    # Validate shape
    expected_shape = (30, model.input_shape[-1])
    if X_test.shape[1:] != expected_shape:
        st.error("❌ Sequence shape mismatch")
        st.info(f"Expected: (batch, {expected_shape})")
        st.info(f"Got: {X_test.shape}")
        return

    if len(df_meta) != X_test.shape[0]:
        st.error("❌ Metadata row count must match number of sequences")
        return

    # Predict
    with st.spinner("🔮 Predicting RUL..."):
        preds = model.predict(X_test, verbose=0).flatten()

    df_meta["RUL"] = preds

    st.success(
        f"Prediction Complete | "
        f"Min: {preds.min():.1f} | "
        f"Max: {preds.max():.1f} | "
        f"Mean: {preds.mean():.1f}"
    )

    st.markdown("---")
    plot_rul_trends(df_meta)
    st.markdown("---")
    render_dashboard(df_meta)
    st.markdown("---")
    plot_alert_zone_counts(df_meta)

# ==============================
# Run App
# ==============================
if __name__ == "__main__":
    main()
