# Google Contacts setup (one-time, ~10 minutes)

This connects `google_contacts.py` to your real Google Contacts (saved
Contacts + Gmail's "Other contacts") so Chloe can resolve a name like
"Madison Wayne" to an email address on her own. Nothing here lets Chloe
*write* to your Contacts — the scopes are read-only.

You already did the equivalent of this once for YouTube playlists, so
step 1 may already be done.

## 1. Google Cloud project + enable the People API

If you already have a Google Cloud project from the YouTube setup, reuse
it — skip to enabling the API. Otherwise:

1. Go to <https://console.cloud.google.com/>, create a project (any name,
   e.g. "Chloe").
2. In that project, go to **APIs & Services → Library**, search for
   **"People API"**, click it, click **Enable**.

## 2. OAuth consent screen

If you already configured this for YouTube, skip this step — the same
consent screen covers both.

1. **APIs & Services → OAuth consent screen**.
2. User type: **External** (unless you have a Google Workspace org and
   want Internal). App name/support email: anything, this is just for you.
3. Add your own Google account under **Test users** (required while the
   app is in "Testing" mode, which is fine — you don't need to publish
   or verify it for personal use).

## 3. OAuth client (Desktop app)

If you already have a Desktop-app OAuth client (from YouTube), you can
reuse the same downloaded JSON file — just copy it to the new filename
in step 4 below instead of creating a new client.

Otherwise:

1. **APIs & Services → Credentials → Create Credentials → OAuth client ID**.
2. Application type: **Desktop app**. Name: anything (e.g. "Chloe Contacts").
3. Click **Create**, then **Download JSON** on the client you just made.

## 4. Save the credentials file

Save (or copy, if reusing the YouTube one) the downloaded JSON as:

```
C:\Chloe\secrets\google_contacts_client_secret.json
```

## 5. Install dependencies (probably already done)

`google-auth-oauthlib` and `google-api-python-client` are already in
`requirements.txt` (added for the YouTube integration). If you haven't
installed them yet:

```
pip install google-auth-oauthlib google-api-python-client
```

## 6. Run the one-time consent flow

From the `jarvis` folder:

```
python google_contacts.py --auth
```

This opens your browser once, asks you to sign in and approve read-only
access to your Contacts, then caches a token to
`C:\Chloe\secrets\google_contacts_oauth_token.json`. You won't be
prompted again — it refreshes itself silently after this.

## 7. Verify it worked

```
python google_contacts.py --list
python google_contacts.py --resolve "Madison Wayne"
```

`--list` should print your contacts; `--resolve` should print the email
address it found for that name (or an honest "no unambiguous match" if
it isn't in your Contacts or Gmail history).

## That's it

Nothing else to wire up — `email_client.resolve_contact()` already
falls back to this automatically whenever the fast local
`email_contacts.json` file doesn't have a name, and caches a hit there
for next time. Just say "send an email to Madison Wayne" and it should
resolve on its own. If it ever says it doesn't know who that is, check
`python google_contacts.py --resolve "..."` directly — it'll tell you
whether the miss is "not connected yet," "no match," or "ambiguous
(multiple people with that name)."
