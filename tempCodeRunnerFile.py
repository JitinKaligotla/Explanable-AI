import matplotlib.pyplot as plt # type: ignore

    # SHAP Explanation
    st.subheader("🔍 Why This Prediction?")
    shap.initjs()

    # Background for SHAP (KMeans fallback)
    background = shap.kmeans(input_df, 1)

    # Explainer using class 1 probability
    explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], background)
    shap_values = explainer.shap_values(input_df)

    # ✅ Summary Plot (bar)
    fig1, ax1 = plt.subplots()
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
    st.pyplot(fig1)

    # ✅ Dependence Plot
    fig2, ax2 = plt.subplots()
    shap.dependence_plot("ApplicantIncome", shap_values, input_df, ax=ax2, show=False)
    st.pyplot(fig2)
