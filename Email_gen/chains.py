import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv


load_dotenv()


class Chain:
    def __init__(self):
        self.llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv('GROQ_API_KEY')# other params...
)
    
    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Vivek, a Physicist with experience in Data Science and Data analytics. 
            He has experience with regression analysis, LLMS, Generative AI. He also has a PhD in particle physics with emphasis on Data analysis.  
            He has good data visualization skills, problem solving skills and analytical skills.
            He  has also worked in Insurance industry for three years and can be a good fit for the job.
            Your job is to write a cold email to the hiring manager named Gerhard Hausmann working at Barmenia Versicherungen, Wuppertal  
            regarding the job mentioned above describing the capability of Vivek in fulfilling their proffesional needs.
            Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}

            Also add how this email is actually generated using langchain groq cloud which demonstrates his capability of using different LLM frameworks. The email has been sent to him automatically by exploiting google api usage which demonstrates his creativity and finding unique solutions. 
            Remember you are Vivek,  
            Do not provide a preamble.
            Also make sure the letter is simple, formal and to the point. Do not add unnecessary flattering adjectives in the letter.
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        
        file_path = "output.txt"
        with open(file_path, 'w') as file:
            file.write(res.content)

        return res.content
    
if __name__ == "__main__":
    print(os.getenv('GROQ_API_KEY') )
    