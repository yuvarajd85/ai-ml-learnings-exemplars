from __future__ import print_function
import base64
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import polars as pl

def get_spam_subjects():
    # Load OAuth credentials (you must generate them once using Google's OAuth flow)
    creds = Credentials.from_authorized_user_file('token.json', ['https://www.googleapis.com/auth/gmail.readonly'])
    service = build('gmail', 'v1', credentials=creds)

    subjects = []
    next_page = None

    while True:
        response = service.users().messages().list(
            userId='me',
            q="in:spam",
            pageToken=next_page
        ).execute()

        if 'messages' not in response:
            break

        for msg in response['messages']:
            full = service.users().messages().get(userId='me', id=msg['id'], format='metadata', metadataHeaders=['Subject']).execute()
            headers = full['payload']['headers']
            subject = next(h['value'] for h in headers if h['name'] == 'Subject')
            subjects.append(subject)

        next_page = response.get('nextPageToken')
        if not next_page:
            break

    return subjects

spam_subject = get_spam_subjects()
print(spam_subject)

spam_dict = [{"subject": sub, "spam_label" : 1} for sub in spam_subject]

df = pl.from_dicts(spam_dict)

print(df.head())

df.write_csv(f"../resources/datasets/daily_spam.csv")