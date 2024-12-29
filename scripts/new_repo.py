import os
import subprocess
import shutil
import requests
import pandas as pd

result = subprocess.check_output("git rev-parse --show-toplevel", shell=True).decode('utf-8')
ROOT = result.strip()

# GitHub API URL for creating repositories
GITHUB_API_URL = "https://api.github.com/user/repos"

# Replace this with your GitHub Personal Access Token
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Function to create a new GitHub repository using a placeholder logic (replace with your logic)
def create_new_github_repo(new_repo_name):
    """
    Creates a new GitHub repository using the GitHub API.

    Parameters:
    - new_repo_name (str): The name of the new GitHub repository.
    """
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Payload for creating a new repository
    data = {
        "name": new_repo_name,  # Name of the new repository
        "description": "We have fixed your repository!",
        "private": False,  # Set to True if you want the repo to be private
        "auto_init": True  # Initialize the repo with an empty README
    }

    # Make the request to GitHub API to create the repository
    response = requests.post(GITHUB_API_URL, json=data, headers=headers)

    if response.status_code == 201:
        print(f"Repository '{new_repo_name}' created successfully.")
    else:
        print(f"Failed to create repository: {response.status_code}")
        print(response.json())


def build_check():
    # FIXME: Replace with actual function logic
    success = True
    fixed = True
    json_data = {}  # Placeholder JSON data
    return success, fixed, json_data

# List of GitHub repositories
filepath = os.path.join(ROOT, "output", "build_check_results.csv")
df = pd.read_csv(filepath)
repos_list = list(df.itertuples(index=False, name=None))

# Iterate over each repository in the list
for repo, status in repos_list:
    # from IPython import embed; embed()
    print(f"Processing repository: {repo}")
    print(f"Status: {status}")

    if status == "Success" or status == "No requirements found":
        continue

    # Extract the repository name from the URL
    repo_name = os.path.basename(repo).replace(".git", "")

    # Extract the username
    username = repo.split("/")[-2]

    # Create a new directory for the repo and move into it

    repo_dir = os.path.join(ROOT, "output", f"{repo_name}_dir")
    os.makedirs(repo_dir, exist_ok=True)
    os.chdir(repo_dir)

    # Clone the GitHub repository
    # FIXME: fails if repo already exists
    subprocess.run(["git", "clone", f"{repo}.git"], check=True)

    # Navigate into the cloned repository folder
    os.chdir(repo_name)
    # if .gitingore exists, add venv/ to it, otherwise create it and add venv/
    if not os.path.exists(".gitignore"):
        os.system("touch .gitignore && echo 'venv/' >> .gitignore")
    else:
        os.system("echo 'venv/' >> .gitignore")

    # Create a virtual environment (using Python's venv module)
    # FIXME: we should care about which python3 version
    subprocess.run(["python3", "-m", "venv", "venv"], check=True)

    # Run the build_check function to fix dependencies or issues
    # FIXME: # Replace with actual function logic
    success, fixed, json_data = build_check()

    # If the build is successful and issues were fixed
    if success and fixed:
        # Move the fixed requirements.txt (if build_check fixed it)
        if os.path.exists("requirements_fixed.txt"):
            shutil.move("requirements_fixed.txt", "requirements.txt")

        # Add all changes, commit, and prepare to reinitialize the repository
        subprocess.run(["git", "add", "*"], check=True)
        subprocess.run(["git", "commit", "-m", "repo fixed your env file"], check=True)

        # Call the function to create a new GitHub repo (this is a placeholder)
        new_repo_name = f"{username}_{repo_name}"
        create_new_github_repo(new_repo_name)

        # Add a new remote to the reinitialized Git repository and push the changes
        new_repo_url = f"git@github.com:grimrepor/{new_repo_name}.git"
        subprocess.run(["git", "remote", "remove", "origin"], check=True)
        subprocess.run(["git", "remote", "add", "origin", new_repo_url], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)

    # After all repository operations, purge pip cache
    subprocess.run(["pip", "cache", "purge"], check=True)
    print("Pip cache purged successfully")

os.chdir(ROOT)
print("All repositories processed successfully.")