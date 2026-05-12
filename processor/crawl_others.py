import pandas as pd
from tqdm import tqdm
from config import *
import openreview
import os


def crawl(conference: str, year: str):

    conference_config = {
        'neurips': {
            'invitation_template': 'NeurIPS.cc/{year}/Conference/-/Submission',
            'decision_template': 'NeurIPS.cc/{year}/Conference/Submission{number}/-/Decision',
            'review_template': 'NeurIPS.cc/{year}/Conference/Submission{number}/-/Official_Review',
            'rating_field': 'rating'
        },
        'icml': {
            'invitation_template': 'ICML.cc/{year}/Conference/-/Submission',
            'decision_template': 'ICML.cc/{year}/Conference/Submission{number}/-/Decision',
            'review_template': 'ICML.cc/{year}/Conference/Submission{number}/-/Official_Review',
            'rating_field': 'overall_recommendation'
        },
    }

    conference_lower = conference.lower()
    config = conference_config[conference_lower]

            
    submission_path = f'../data/{conference_lower}_{year}/submissions/submissions.csv'
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

            
    client = openreview.api.OpenReviewClient(
        baseurl=SEARCH_URL,
        username=SEARCH_USERNAME,
        password=SEARCH_PASSWORD
    )

            
    invitation = config['invitation_template'].format(year=year)
    submissions = client.get_all_notes(invitation=invitation)

          
    data_list = []
    for note in tqdm(submissions, desc="Crawling", total=len(submissions)):
              
        decision_invitation = config['decision_template'].format(year=year, number=note.number)
        decisions = client.get_all_notes(invitation=decision_invitation)
        decision = decisions[0].content.get('decision', {}).get('value', '') if decisions else ''

                
        review_invitation = config['review_template'].format(year=year, number=note.number)
        reviews = client.get_all_notes(invitation=review_invitation)
        ratings = [r.content.get(config['rating_field'], {}).get('value', '') for r in reviews]
        avg_rating = sum([int(str(r).split(':')[0]) for r in ratings if r]) / len(ratings) if ratings else None

        data_list.append({
            'submission_id': note.id,
            'number': note.number,
            'title': note.content.get('title', {}).get('value', ''),
            'abstract': note.content.get('abstract', {}).get('value', ''),
            'decision': decision,
            'ratings': ','.join(map(str, ratings)),
            'avg_rating': avg_rating
        })
        print({
            'submission_id': note.id,
            'number': note.number,
            'title': note.content.get('title', {}).get('value', ''),
            'abstract': note.content.get('abstract', {}).get('value', ''),
            'decision': decision,
            'ratings': ','.join(map(str, ratings)),
            'avg_rating': avg_rating
        })

    df = pd.DataFrame(data_list)
    df.to_csv(submission_path, index=False)

    print(f"Crawl {conference_lower}_{year} finished!")


if __name__ == "__main__":
    crawl("icml", "2025")
    crawl("neurips", "2025")
