import streamlit as st
import tempfile
import sys
from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from backend.pipeline import run_pipeline

st.set_page_config(
    page_title="Insurance Claim Intelligence System",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Insurance Claim Intelligence System")

st.write(
    "Upload a damaged vehicle image and enter the claim description."
)

uploaded_image = st.file_uploader(
    "Upload Vehicle Image",
    type=["jpg", "jpeg", "png"]
)

claim_text = st.text_area(
    "Claim Description",
    height=150
)

analyze = st.button(
    "Analyze Claim",
    use_container_width=True
)

if uploaded_image:

    st.image(
        uploaded_image,
        use_container_width=True
    )

if analyze:

    if uploaded_image is None:
        st.error("Please upload an image.")
        st.stop()

    if not claim_text.strip():
        st.error("Please enter a claim description.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False,suffix=".jpg") as tmp:
        tmp.write(uploaded_image.read())
        image_path = tmp.name

    

    with st.spinner("Analyzing insurance claim..."):

        result = run_pipeline(image_path=image_path,claim_text=claim_text)

    fusion = result["fusion_result"]

    st.success("Analysis completed successfully!")

    st.subheader("Uploaded Vehicle")
    
    st.image(uploaded_image,width=500)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
        "Image Prediction",
        fusion["image_prediction"]["label"],
        f"{fusion['image_prediction']['confidence']*100:.1f}%"
        )

        st.progress(
        fusion["image_prediction"]["confidence"]
        )

    with col2:
        st.metric(
        "Text Prediction",
        fusion["text_prediction"]["label"],
        f"{fusion['text_prediction']['confidence']*100:.1f}%"
        )

    with col3:
        final_damage = list(fusion["final_damage"].keys())[0]
        final_confidence = list(fusion["final_damage"].values())[0]

        st.metric(
            "Final Damage",
            final_damage,
            f"{final_confidence*100:.1f}%"
        )

        st.progress(final_confidence)
       

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Agreement Score",
            f"{fusion['agreement_score']*100:.1f}%"
        )

    with col2:
        st.metric(
            "Consistency",
            fusion["consistency"]
        )

    with col3:
        st.metric(
            "Conflict Level",
            fusion["conflict_level"]
        )

    st.subheader("Retrieved Policy Evidence")

    with st.expander("Retrieved Policy Evidence"):
        for source in result["sources"]:
            st.write(
                f"📄 {source['source']}  |  Page {source['page']}"
            )

    report = result["report"]

    st.subheader("Executive Summary")

    st.info(report["executive_summary"])

    action = fusion["recommended_action"]

    st.subheader("Recommended Action")

    if action == "Auto Approve" or action == "Approve":
        st.success(action)

    elif action == "Manual Verification":
        st.warning(action)

    else:
        st.error(action)

    st.subheader("Next Steps")

    for step in report["next_steps"]:

        st.checkbox(step, value=False)


    st.download_button(
        "Download JSON Report",
        data=json.dumps(report, indent=4),
        file_name="claim_report.json",
        mime="application/json"
    )