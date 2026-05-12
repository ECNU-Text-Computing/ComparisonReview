import pandas as pd
from config import *
import openreview


def crawl(venue: str):

    submission_path = f'../data/{venue}/submissions/submissions.csv'

    client = openreview.api.OpenReviewClient(
        baseurl=SEARCH_URL,
        username=SEARCH_USERNAME,
        password=SEARCH_PASSWORD
    )

             
    submissions = client.get_all_notes(
        invitation=f'ICLR.cc/2025/Conference/-/Submission'
    )
          
    data_list = []
    for note in submissions:
        data_list.append({
            'submission_id': note.id,
            'title': note.content.get('title', {}).get('value', ''),
            'abstract': note.content.get('abstract', {}).get('value', '')
        })
    df = pd.DataFrame(data_list)
    df.to_csv(submission_path, index=False)

    print(f"Crawl {venue} finished!")


if __name__ == "__main__":
    crawl("iclr_2025")


