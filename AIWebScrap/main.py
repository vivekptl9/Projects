import streamlit as st
from scrape import scrape_website, extract_body_content,split_dom_content,clean_body_content

st.title("AI Web Scraper")
url = st.text_input("Enter a Website URL: ")


if st.button("Scrape Site"):
    st.write("Scraping the website")
    result = scrape_website(url)


    body_content =extract_body_content(result)
    clean_content=clean_body_content(body_content)

    st.session_state.dom_content = clean_content
    with st.expander("View DOM Content"):
        st.text_area("Dom Content", clean_content, height=300)


    


    