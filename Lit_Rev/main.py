import arxiv
from transformers import pipeline

def fetch_arxiv_papers(query, max_results=5):
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        papers = []
        for result in search.results():
            papers.append({
                "title": result.title,
                "summary": result.summary,
                "url": result.entry_id
            })
        if not papers:
            print(f"No papers found for query: {query}")
        return papers
    except Exception as e:
        print(f"Error fetching papers: {e}")
        return []

# Example query: Particle Physics
papers = fetch_arxiv_papers("particle physics", max_results=3)

# Check if papers were fetched and output
if papers:
    for paper in papers:
        print(f"Title: {paper['title']}\nLink: {paper['url']}\n")
else:
    print("No papers found or an error occurred.")




# Initialize Hugging Face's summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Summarize the fetched papers
def summarize_paper(paper_summary):
    # Hugging Face models have a max token limit, so summarization may need chunking
    summary = summarizer(paper_summary, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']

# Summarize each paper
for paper in papers:
    summary = summarize_paper(paper['summary'])
    print(f"Paper: {paper['title']}\nSummary: {summary}\nLink: {paper['url']}\n")

# Initialize a Hugging Face question-answering model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def extract_key_findings(paper_summary):
    # Asking a specific question to extract insights
    question = "What are the key findings of this research?"
    result = qa_pipeline(question=question, context=paper_summary)
    return result['answer']

# Extract key findings from each paper
for paper in papers:
    findings = extract_key_findings(paper['summary'])
    print(f"Paper: {paper['title']}\nKey Findings: {findings}\nLink: {paper['url']}\n")




