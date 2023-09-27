import os
import streamlit as st
import subprocess

def check_password(username, password):
    user_data = load_user_data()
    if username in user_data:
        stored_password = user_data[username]
        return password == stored_password
    return False

def load_user_data():
    user_data = {}
    if os.path.isfile("user_data.txt"):
        with open("user_data.txt", "r") as file:
            for line in file:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    username, stored_password = parts
                    user_data[username] = stored_password
    return user_data

def save_user_data(user_data):
    with open("user_data.txt", "w") as file:
        for username, stored_password in user_data.items():
            file.write(f"{username}:{stored_password}\n")

selected_option = st.sidebar.radio("Select Option", ("Login", "Signup"))

username = ""
password = ""

if selected_option == "Signup":
    st.title("User Signup")
    
    username = st.text_input("Username", value=username)  
    password = st.text_input("Password", type="password")
    
    if st.button("Register"):
        user_data = load_user_data()
        if username in user_data:
            st.error("Username already exists. Please choose a different one.")
           
        else:
            plain_text_password = password

            user_data[username] = plain_text_password
            save_user_data(user_data)

            st.success("Registration successful. You can now login.")
            st.session_state["username"] = ""

elif selected_option == "Login":
    st.title("User Login")
    username = st.text_input("Username", value=username)  
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if check_password(username, password):
            st.success(f"Welcome, {username}! You are now logged in.")
            
            
            subprocess.Popen(["streamlit", "run", "https://wine-recommendation.streamlit.app/"])
        else:
            st.error("Invalid username or password. Please try again.")
           

