import os
import pandas as pd
from bs4 import BeautifulSoup


def get_df(
        venue: str = "naacl_2025",
        subset: str = "findings",
        number: int = 250,
        random_seed: int = 42
):
              
    with open(f'../data/acl_anthology/{venue}_{subset}.html', 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

            
    papers = []
    for paper in soup.find_all('p', class_='d-sm-flex align-items-stretch'):
              
        title_tag = paper.find('strong').find('a')
        if title_tag:
            title = title_tag.get_text(strip=True)
                  
            abstract = None
            abstract_id = paper.find('a', {'data-toggle': 'collapse'})
            if abstract_id:
                abstract_div_id = abstract_id.get('href', '').replace('#', '')
                abstract_div = soup.find('div', id=abstract_div_id)
                if abstract_div:
                    abstract = abstract_div.get_text(strip=True)
            papers.append({'title': title, 'abstract': abstract})

                 
    df = pd.DataFrame(papers)
    df = df.dropna()
                                 
                      
    df = df.sample(n=number, random_state=random_seed)
    return df


def main(venue):
    findings_df = get_df(venue=venue, subset='findings', number=250)
    long_df = get_df(venue=venue, subset='long', number=250)

    findings_df['avg_rating'] = 0.0
    findings_df['decision'] = 'Reject'

    long_df['avg_rating'] = 10.0
    long_df['decision'] = 'Accept'

    df = pd.concat([findings_df, long_df], ignore_index=True)

    print(df)
    if not os.path.exists(f'../data/{venue}/submissions'):
        os.makedirs(f'../data/{venue}/submissions')
    df.to_csv(f'../data/{venue}/submissions/submissions.csv', index=False)


if __name__ == '__main__':
    main("acl_2025")
    main("naacl_2025")
    main("emnlp_2025")

