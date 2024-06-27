import pandas as pd
from flask import Flask, render_template, request
import pickle

# TO EDIT

FILES_PATH = '/files'
PAPER_MODEL_PATH = f'{FILES_PATH}/l2rRanker_fitted.pickle'
PAPER_DATA_PATH = f'{FILES_PATH}/paper_author_org/paper_level_edited.csv'
AUTHOR_MODEL_PATH = f'{FILES_PATH}/author_l2rRanker_fitted.pickle'
AUTHOR_DATA_PATH = f'{FILES_PATH}/paper_author_org/author_level_edited.csv'
query_number = 15      # show top 10

print("Loading paper data:")
paper_data_df = pd.read_csv(PAPER_DATA_PATH)
print("Done")

print("Loading paper model:")
with open(PAPER_MODEL_PATH, 'rb') as f:
    l2rRanker = pickle.load(f)
print("Done")

print("Loading author data:")
author_data_df = pd.read_csv(AUTHOR_DATA_PATH)
print("Done")

print("Loading author model:")
with open(AUTHOR_MODEL_PATH, 'rb') as f:
    author_l2rRanker = pickle.load(f)
print("Done")


print("Initializing app:")
app = Flask(__name__)

# authors_df = 'dataset/dummy_author_data.csv'
# papers_df = 'dataset/dummy_paper_data.csv'
# orgs_df = 'dataset/dummy_org_data.csv'

# global_paper_data = pd.read_csv(papers_df)
# global_author_data = pd.read_csv(authors_df)
# global_org_data = pd.read_csv(orgs_df)

@app.route("/", methods=("GET", "POST"))
def index():
    # Default value
    search_type = 'author'  
    result_data = None

    if request.method == "POST":
        # This will be either 'author', 'paper', or 'org'
        search_type = request.form['searchType']  
        query = request.form.get("query")
        result_data = search(query, search_type)
    
    return render_template("index.html", result=result_data, searchType=search_type)


def search(query, search_type):
    
    if search_type == 'paper':
        # Group by title and aggregate fields
        rank_result_list = l2rRanker.query(query)
        rank_result_df = pd.DataFrame(rank_result_list, columns=['docid','score'])
        rank_result_df = pd.merge(paper_data_df, rank_result_df)
        rank_result_df = rank_result_df.sort_values('score', ascending=False)[:query_number].drop(columns='score')
        rank_result_df = rank_result_df.fillna('-')
        rank_result_df['org'] = rank_result_df['org'].apply(lambda x: "\n".join(set(x.split('; '))))
        rank_result_df['author'] = rank_result_df['author'].apply(lambda x: "\n".join(set(x.split('; '))))
        
        # result = rank_result_df.groupby('title').agg({
        #     'abstract': 'first',  # Abstract is the same for each title
        #     'year': 'first',      # Year is the same for each title
        #     'author': '\n'.join,  # List of all authors
        #     'org': lambda x: '\n'.join(list(set(x))),  # List of all organizations
        #     'total_citations': 'first'    # Citation count is the same for each title
        # }).reset_index().rename(columns={'total_citations': 'n_citation'})
        return rank_result_df[["title", "abstract", "year", "author", "org", "n_citation"]]

    elif search_type == 'author':
        
        rank_result_list = author_l2rRanker.query(query)
        rank_result_df = pd.DataFrame(rank_result_list, columns=['authorid','score'])
        rank_result_df = pd.merge(author_data_df, rank_result_df)
        rank_result_df = rank_result_df.sort_values('score', ascending=False)[:query_number].drop(columns='score')
        rank_result_df = rank_result_df.fillna('-')
        rank_result_df = rank_result_df.rename(columns={'n_citations': 'n_citation', 'serpapi_link': 'google_link'})
        
#         result = global_author_data[['author', 'org', 'n_citation', 'google_link']]
        return rank_result_df[['author', 'org', 'n_citation', 'google_link']]

    elif search_type == 'org':
        
        rank_result_list = author_l2rRanker.query(query)
        rank_result_df = pd.DataFrame(rank_result_list, columns=['authorid','score'])
        rank_result_df = pd.merge(author_data_df, rank_result_df)
        rank_result_df = rank_result_df.sort_values('score', ascending=False)[:1000].drop(columns='score')
        rank_result_df = rank_result_df[['org', 'n_citations']].groupby('org').sum()
        rank_result_df = rank_result_df.reset_index().sort_values('n_citations', ascending=False)[:query_number]
        rank_result_df = rank_result_df.rename(columns={'n_citations': 'n_citation'})
        
        # result = global_org_data[['org', 'n_citation']]

        return rank_result_df

if __name__ == '__main__':
    app.run()
