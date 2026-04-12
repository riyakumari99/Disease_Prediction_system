import streamlit as st
import pickle, json, pandas as pd, os, datetime
import matplotlib.pyplot as plt

# ================= LOAD MODEL =================
model = pickle.load(open("model_nb.pkl", "rb"))
le = pickle.load(open("label_encoder_nb.pkl", "rb"))
columns = json.load(open("columns_nb.json", "r"))

desc_dict = pickle.load(open("desc_nb.pkl", "rb"))
prec_dict = pickle.load(open("prec_nb.pkl", "rb"))

col_index = {col: i for i, col in enumerate(columns)}

# ================= FILE INIT =================
for file in ["users.json", "history.json"]:
    if not os.path.exists(file):
        with open(file, "w") as f:
            json.dump({}, f)

with open("users.json", "r") as f:
    users = json.load(f)

with open("history.json", "r") as f:
    history = json.load(f)

# ================= SESSION =================
if "user" not in st.session_state:
    st.session_state.user = None

# ================= SIDEBAR LOGIN =================
st.sidebar.title("🔐 Account")

menu = st.sidebar.radio("Menu", ["Login", "Signup"])

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if menu == "Signup":
    if st.sidebar.button("Create Account"):
        if username in users:
            st.sidebar.error("User exists")
        else:
            users[username] = password
            json.dump(users, open("users.json", "w"))
            st.sidebar.success("Account created")

if menu == "Login":
    if st.sidebar.button("Login"):
        if username in users and users[username] == password:
            st.session_state.user = username
            st.sidebar.success(f"Welcome {username}")
        else:
            st.sidebar.error("Invalid login")

if st.session_state.user:
    st.sidebar.success(f"Logged in as {st.session_state.user}")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()

# ================= MAIN =================
st.title("🩺 AI Disease Prediction System")

if st.session_state.user:

    page = st.sidebar.selectbox(
        "Navigation",
        ["Prediction", "History", "Chatbot", "Hospitals"]
    )

    # ================= PREDICTION =================
    if page == "Prediction":

        st.subheader("🔍 Select Symptoms")
        selected = st.multiselect("Symptoms", columns)

        if st.button("Predict"):
            if not selected:
                st.warning("Select symptoms")
            else:
                input_data = [0]*len(columns)

                for s in selected:
                    input_data[col_index[s]] = 1

                probs = model.predict_proba(
                    pd.DataFrame([input_data], columns=columns)
                )[0]

                top3 = probs.argsort()[-3:][::-1]

                st.subheader("Results")

                results = []

                doctor_map = {
                    "Dengue": "General Physician",
                    "Diabetes": "Endocrinologist",
                    "Heart attack": "Cardiologist",
                    "Migraine": "Neurologist"
                }

                for i in top3:
                    disease = le.classes_[i]
                    prob = round(probs[i]*100,2)

                    results.append({"disease": disease, "prob": prob})

                    st.markdown(f"### {disease}")
                    st.progress(int(prob))
                    st.write(f"{prob}%")

                    st.info(desc_dict.get(disease,""))

                    for p in prec_dict.get(disease, []):
                        st.write("✔", p)

                    st.success("Doctor: " + doctor_map.get(disease,"General Physician"))

                    st.markdown("---")

                # GRAPH
                labels = [r["disease"] for r in results]
                values = [r["prob"] for r in results]

                fig, ax = plt.subplots()
                ax.barh(labels, values)
                st.pyplot(fig)

                # SAVE HISTORY
                user = st.session_state.user
                if user not in history:
                    history[user] = []

                history[user].append({
                    "symptoms": selected,
                    "results": results,
                    "time": str(datetime.datetime.now())
                })

                json.dump(history, open("history.json","w"), indent=4)

                st.success("Saved!")

    # ================= HISTORY =================
    elif page == "History":

        user = st.session_state.user
        user_hist = history.get(user, [])

        if not user_hist:
            st.info("No history")
        else:
            for h in reversed(user_hist):
                st.write("Time:", h["time"])
                st.write("Symptoms:", h["symptoms"])
                st.write("Prediction:", h["results"][0]["disease"])
                st.markdown("---")

    # ================= CHATBOT (FREE) =================
    elif page == "Chatbot":

        st.subheader("🤖 Health Assistant")

        user_input = st.text_input("Ask something")

        if st.button("Ask"):

            # SIMPLE RULE-BASED CHATBOT
            if "fever" in user_input.lower():
                reply = "You may have infection. Drink fluids and consult doctor."
            elif "headache" in user_input.lower():
                reply = "Take rest, stay hydrated. If severe, consult doctor."
            elif "cold" in user_input.lower():
                reply = "Common cold. Take steam and warm fluids."
            else:
                reply = "Please consult a doctor for proper diagnosis."

            st.write("🤖:", reply)

    # ================= HOSPITALS =================
    elif page == "Hospitals":

        st.subheader("🏥 Nearby Hospitals")

        location = st.text_input("Enter city", "Bathinda")

        if location:
            map_url = f"https://www.google.com/maps?q=hospitals+near+{location}&output=embed"
            st.components.v1.iframe(map_url, height=400)

else:
    st.warning("Login first")