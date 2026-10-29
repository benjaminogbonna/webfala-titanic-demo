import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")


# Load the trained model once and cache it across reruns/sessions
@st.cache_resource
def load_model():
    return joblib.load("models/forest_model.joblib")


random_forest = load_model()


# THE FUNCTION
def predict_survival(name, Pclass, Sex, Age, SibSp, PaCh, Fare):
    input_data = pd.DataFrame({
        "Pclass": [Pclass],
        "Sex": [Sex],
        "Age": [Age],
        "SibSp": [SibSp],
        "PaCh": [PaCh],
        "Fare": [Fare],
    })

    prediction = random_forest.predict(input_data)[0]
    probabilities = random_forest.predict_proba(input_data)[0]

    name = name.strip().title() or "Passenger"
    if prediction == 0:
        result = f"Wooo 😮 {name}, you Died 😢"
    else:
        result = f"Cheers 🥂 {name}, you Survived 😁"

    # probs = {"Died": float(probabilities[0]), "Survived": float(probabilities[1])}
    return prediction, result, probabilities


DEFAULTS = {
    "name": "",
    "pclass": 3,
    "sex": 0,
    "age": 30,
    "sibsp": 0,
    "pach": 0,
    "fare": 30.0,
}

# Initialise widget state once
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def load_example(example):
    name, pclass, sex, age, sibsp, pach, fare = example
    st.session_state.update(
        name=name, pclass=pclass, sex=sex, age=age,
        sibsp=sibsp, pach=pach, fare=float(fare),
    )


# THE UI
st.title("Titanic Survival Predictor")
st.write("Enter a passenger's details and see if our random forest thinks they'd survive.")

with st.form("passenger_form"):
    st.text_input("Name", placeholder="e.g. Benjamin Ogbonna", key="name")
    st.radio(
        "Passenger Class (1 = First)", [1, 2, 3],
        horizontal=True, key="pclass",
    )
    st.radio(
        "Sex", [0, 1],
        format_func=lambda x: "Male" if x == 0 else "Female",
        horizontal=True, key="sex",
    )
    st.slider("Age", 0, 80, step=1, key="age")

    c1, c2, c3 = st.columns(3)
    c1.number_input("Siblings/Spouses aboard", min_value=0, step=1, key="sibsp")
    c2.number_input("Parents/Children aboard", min_value=0, step=1, key="pach")
    c3.number_input("Fare paid (£)", min_value=0.0, step=1.0, key="fare")

    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    s = st.session_state
    prediction, result, probabilities = predict_survival(
        s.name, s.pclass, s.sex, s.age, s.sibsp, s.pach, s.fare
    )

    st.subheader("Prediction")
    if prediction == 0:
        st.error(result)
    else:
        st.success(result)

    st.subheader("Probability")
    if prediction == 0:
        st.write(f"{probabilities[0]:.1%}")
    else:
        st.write(f"{probabilities[1]:.1%}")
  