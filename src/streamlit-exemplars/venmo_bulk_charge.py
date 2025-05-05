'''
Created on 5/4/2025 at 10:38 PM
By yuvaraj
Module Name: venmo_bulk_charge
'''
import streamlit as st
import pandas as pd
import urllib.parse
import webbrowser

st.set_page_config(page_title="Venmo Bulk Charger", layout="centered")
st.title("💸 Venmo Bulk Charge Request Tool")

st.markdown("Upload a CSV file or enter usernames manually to generate Venmo charge links.")

# Step 1: Upload CSV or manual entry
usernames = set()

uploaded_file = st.file_uploader("Upload CSV with 'username' column", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if 'Username' in df.columns:
            usernames.update(df['Username'].dropna().astype(str).str.strip().tolist())
            st.success(f"Loaded {len(usernames)} usernames from file.")
        else:
            st.error("CSV must contain a 'username' column.")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

manual_input = st.text_area("Or enter additional usernames (one per line):")
if manual_input:
    usernames.update([u.strip() for u in manual_input.splitlines() if u.strip()])

if not usernames:
    st.warning("Please upload a CSV or enter usernames manually to proceed.")

# Show multiselect of usernames
selected_usernames = st.multiselect("Select users to charge", sorted(usernames), default=sorted(usernames))

# Step 2: Enter charge info
amount = st.number_input("💵 Charge Amount ($)", min_value=0.01, step=0.01, format="%.2f")
note = st.text_input("📝 Charge Note", value="Shared expense")

# Step 3: Generate Links
if st.button("🔗 Generate Venmo Charge Links"):
    if not selected_usernames:
        st.warning("Please select at least one user.")
    else:
        st.success(f"Generated links for {len(selected_usernames)} users")
        for user in selected_usernames:
            params = {
                "txn": "charge",  # or "pay"
                "amount": amount,
                "note": note
            }
            query = urllib.parse.urlencode(params)
            link = f"https://venmo.com/{user}?{query}"
            st.markdown(f"👉 [Charge @{user}]({link})", unsafe_allow_html=True)

# Optional: Open in browser
if st.button("🌐 Open All in Browser (Caution)"):
    for user in selected_usernames:
        params = {
            "txn": "charge",
            "amount": amount,
            "note": note
        }
        query = urllib.parse.urlencode(params)
        link = f"https://venmo.com/{user}?{query}"
        webbrowser.open_new_tab(link)

