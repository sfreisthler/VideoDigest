import streamlit as st

# --- Page Settings ---
page_icon = "🎬"
page_title = "VideoDigest"
layout = "centered"

st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + " " + page_icon)

# --- Hide Streamlit Style ---
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- Get Data From User --- 
with st.form("entry_form", clear_on_submit=True):
    video_url = st.text_input("Video URL:")
    submitted = st.form_submit_button("Search Video")

if submitted:
    question = st.text_input("Question: ")