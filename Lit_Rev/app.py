import streamlit as st
import arxiv
import requests
import os

# Function to search for papers based on text input and filter by date manually
def search_arxiv(query, start_year, end_year, max_results):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    # Filter papers based on the year of publication
    filtered_papers = [
        paper for paper in search.results()
        if start_year <= paper.published.year <= end_year
    ]
    return filtered_papers

# Function to download a paper from arxiv link and save locally
def download_arxiv_paper(arxiv_link, save_path):
    arxiv_id = arxiv_link.split('/')[-1]
    paper_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    response = requests.get(paper_url)
    
    if response.status_code == 200:
        # Save PDF locally
        with open(f"{save_path}/{arxiv_id}.pdf", "wb") as f:
            f.write(response.content)
        return f"{save_path}/{arxiv_id}.pdf"
    else:
        st.error("Failed to download the paper. Please check the link.")
        return None

# Streamlit UI
st.title("Arxiv Paper Downloader")

# Specify directory to save the files (current directory by default)
save_dir = os.getcwd()

# Tab for paper search or direct download via link
search_type = st.radio("Choose an option:", ("Search by Text", "Download by Arxiv Link"))

if search_type == "Search by Text":
    # Text search input
    query = st.text_input("Enter the search term:")
    
    # Select year range and number of papers
    year_range = st.slider("Select publication year range:", 1990, 2024, (2010, 2024))
    num_papers = st.number_input("Number of papers to download:", min_value=1, max_value=100, value=10)
    
    if st.button("Search and Download Papers"):
        if query:
            with st.spinner("Searching papers..."):
                papers = search_arxiv(query, year_range[0], year_range[1], num_papers)
            
            if papers:
                st.success(f"Found {len(papers)} papers.")
                progress = st.progress(0)
                
                for idx, paper in enumerate(papers):
                    st.write(f"**Title**: {paper.title}")
                    st.write(f"**Authors**: {', '.join([author.name for author in paper.authors])}")
                    st.write(f"**Published**: {paper.published}")
                    
                    # Save PDF locally and provide a download button
                    pdf_local_path = download_arxiv_paper(paper.pdf_url, save_dir)
                    if pdf_local_path:
                        st.success(f"Downloaded paper: {paper.title}")
                        st.write(f"Saved at: {pdf_local_path}")
                        with open(pdf_local_path, "rb") as pdf_file:
                            st.download_button(
                                label="Download PDF",
                                data=pdf_file,
                                file_name=f"{paper.title}.pdf"
                            )
                    
                    # Update progress bar
                    progress.progress((idx + 1) / len(papers))
            else:
                st.error("No papers found for the given search term.")
        else:
            st.error("Please enter a search term.")

elif search_type == "Download by Arxiv Link":
    arxiv_link = st.text_input("Paste the Arxiv link:")
    
    if st.button("Download"):
        if arxiv_link:
            with st.spinner("Downloading paper..."):
                pdf_local_path = download_arxiv_paper(arxiv_link, save_dir)
                
            if pdf_local_path:
                arxiv_id = arxiv_link.split('/')[-1]
                st.success(f"Paper with Arxiv ID {arxiv_id} downloaded.")
                st.write(f"Saved at: {pdf_local_path}")
                with open(pdf_local_path, "rb") as pdf_file:
                    st.download_button(
                        label="Download PDF",
                        data=pdf_file,
                        file_name=f"arxiv_{arxiv_id}.pdf"
                    )
        else:
            st.error("Please paste a valid Arxiv link.")
