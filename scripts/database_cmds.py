import os
import sys
import re
import subprocess
import pandas as pd
import requests
import json
import mysqlx
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()
github_token = os.getenv("GITHUB_TOKEN")
# github_token = os.getenv("GITHUB_ACCESS_TOKEN")

ROOT = subprocess.check_output("git rev-parse --show-toplevel", shell=True).decode('utf-8').strip()
OS = sys.platform
if OS != 'linux' and OS != 'darwin':
    raise Exception('Unsupported OS')


def install_mysql() -> bool:
    """
    choosing default installation (not secure)
    run mysql_secure_installation for secure installation
    """
    # if already installed, return
    cmd_test = ['mysql', '--version']

    if OS == "linux":
        cmd_test.insert(0, "sudo")
        result = subprocess.run(cmd_test, check=True)
        if result.returncode == 0:
            print("MySQL is already installed.")
            return True
        else:
            print("MySQL is not installed. Installing MySQL...")
            # install mysql if not installed and check status to verify
            try:
                # subprocess.run(['sudo', 'apt', 'update'], check=True)
                subprocess.run(['sudo', 'apt', 'install', '-y', 'mysql-server'], check=True)
                subprocess.run(['sudo', 'systemctl', 'status', 'mysql'], check=True)
                print("MySQL installation completed.")
                return True
            except Exception as e:
                print(f"Error installing mysql: {str(e)}")
                return False
    elif OS == 'darwin':
        result = subprocess.run(cmd_test)
        if result.returncode == 0:
            print("MySQL is already installed.")
            return True
        else:
            print("MySQL is not installed. Installing MySQL...")
            try:
                subprocess.run(['brew', 'install', 'mysql'], check=True)
                print("MySQL installation completed.")
                return True
            except Exception as e:
                print(f"Error installing mysql: {str(e)}")
                return False
    else:
        print("Unsupported OS")
        return False

def launch_server() -> bool:
    if OS == "linux":
        try:
            subprocess.run(['sudo', 'systemctl', 'start', 'mysql'], check=True)
            print("MySQL server started successfully on Linux.\n")
            return True
        except Exception as e:
            print(f"Error starting MySQL server on Linux: {str(e)}")
            return False
    elif OS == 'darwin':
        try:
            subprocess.run(['brew', 'services', 'start', 'mysql'], check=True)
            print("MySQL server started successfully on macOS.\n")
            return True
        except Exception as e:
            print(f"Error starting MySQL server on macOS: {str(e)}")
            return False
    else:
        print("Unsupported OS")
        return False

def spinup_mysql_server() -> bool:
    if not install_mysql():
        print("Error installing mysql")
        return False

    if not launch_server():
        print("Error launching mysql server")
        return False

    return True

def create_session(db_name: str = None) -> object:
    """
    create mysql server session
    can create a database without giving db_name
    and call later with db_name to connect to the database
    returns the session object (open connection)
    """
    conn_params = {}
    conn_params["host"] = str(os.getenv("MYSQL_HOST", "localhost"))
    conn_params["port"] = int(os.getenv("MYSQL_PORT", "33060"))
    conn_params["user"] = str(os.getenv("MYSQL_USER", "root"))
    conn_params["password"] = str(os.getenv("MYSQL_PASSWORD", ""))

    try:
        session = mysqlx.get_session(**conn_params)
        schema = None
        if db_name:
            schema = session.get_schema(db_name)
            # WARNING: we intentionally select the database for the session
            # this will propogate to chained functions but not to new sessions
            session.sql(f"USE {db_name}").execute()  # Ensure the database is selected

        # print(f'{db_name = }\t{session = }\t{schema = }')
        return session, schema
    except Exception as e:
        print(f"Error connecting to mysql as '{conn_params['user']}'@'{conn_params['host']}'\n{str(e)}")
        # return None, None
        sys.exit(1)

def create_db(db_name: str = "grimrepor_db") -> bool:
    """
    create a new database
    ok if the database already exists
    """
    session, _ = create_session()
    try:
        session.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}").execute()
        print(f"Database '{db_name}' is active.")
    except mysqlx.DatabaseError as e:
        if "schema exists" in str(e).lower():
            print(f"Database {db_name} already exists.")
            return True
    except Exception as e:
        print(f"Error creating database: {str(e)}")
        return False
    finally:
        session.close()

def show_databases() -> bool:
    session, _ = create_session()
    try:
        print(f"\nDatabases: {session.get_schemas()}\n")
        if session:
            session.close()
        return True
    except Exception as e:
        print(f"Error showing databases: {str(e)}")
        return False
    finally:
        if session: session.close()

def show_all_tables(db_name: str = "grimrepor_db") -> bool:
    """
    show all tables in the database
    """
    session, schema = create_session(db_name)
    if not session: return False
    try:
        print(f"Tables in {db_name = }\n")
        _ = [ print(f'  {table.get_name()}') for table in schema.get_tables() ]
        print()
        session.close()
        return True
    except Exception as e:
        print(f"Error showing tables: {str(e)}")
        return False
    finally:
        session.close()

def show_table_columns(table_name: str, db_name: str = "grimrepor_db") -> bool:
    session, _ = create_session(db_name)
    if not session: return False
    print(f"\nColumns in table '{table_name}'\n")
    try:
        result = session.sql(f"SHOW COLUMNS FROM {table_name}").execute()
        for col in result.fetch_all():
            print(f'  {col}')
        print()
        return True
    except Exception as e:
        print(f"Error showing table columns: {str(e)}")
        return False
    finally:
        session.close()

def show_table_contents(table_name: str, db_name: str = "grimrepor_db", limit_num: int = None) -> bool:
    """
    SELECT * FROM table_name;
    option to limit the number of rows returned
    shows all columns
    """
    session, schema = create_session(db_name)
    if not session: return False

    try:
        print(f"Contents of {table_name}\n")
        result = None
        try:
            if limit_num is None:
                result = schema.get_table(table_name).select().execute()
            else:
                result = schema.get_table(table_name).select().limit(limit_num).execute()
        except Exception as e:
            print(f"Error selecting from table: {str(e)}")
            return False

        for row in result.fetch_all():
            print(row)
        return True

    except Exception as e:
        print(f"Error showing table contents: {str(e)}")
        return False
    finally:
        session.close()

def drop_table(table_name: str, db_name: str = "grimrepor_db") -> bool:
    """
    drop a table from the database
    be careful with this command as it will delete a table
    """
    session, _ = create_session(db_name)
    try:
        session.sql("SET FOREIGN_KEY_CHECKS = 0").execute()  # Disable foreign key checks
        session.sql(f"DROP TABLE IF EXISTS {table_name}").execute()
        print(f"Table {table_name} dropped successfully.")
        return True
        session.sql("SET FOREIGN_KEY_CHECKS = 1").execute()  # Re-enable foreign key checks
    except Exception as e:
        print(f"Error dropping table: {str(e)}")
        return False
    finally:
        if session: session.close()

def drop_all_tables(db_name: str = "grimrepor_db") -> bool:
    """
    drop all of the tables
    find all the tables in the database
    then drop them one by one
    """
    ans = input("Are you sure you want to drop all tables? (y/n): ")
    if ans.lower() != 'y':
        print("Tables not dropped.")
        return False

    session, schema = create_session(db_name)
    if not session: return False

    try:
        tables = schema.get_tables()
        session.sql("SET FOREIGN_KEY_CHECKS = 0").execute()  # Disable foreign key checks
        for table in tables:
            table_name = table.get_name()
            session.sql(f"DROP TABLE IF EXISTS {table_name}").execute()
            print(f"Table {table_name} dropped successfully.")
        session.sql("SET FOREIGN_KEY_CHECKS = 1").execute()  # Re-enable foreign key checks
        return True
    except Exception as e:
        print(f"Error dropping tables: {str(e)}")
        return False
    finally:
        session.close()

def delete_data_from_table(table_name: str, db_name: str = "grimrepor_db") -> bool:
    """
    delete all data from a table
    keep column headers
    """
    session, _ = create_session(db_name)
    if not session: return False

    try:
        session.sql(f"TRUNCATE TABLE {table_name}").execute()
        print(f"Data deleted from table {table_name}.")
        return True
    except Exception as e:
        print(f"Error deleting data from table: {str(e)}")
        return False
    finally:
        session.close()


class Table:
    def __init__(self, table_name: str, db_name: str = "grimrepor_db"):
        self.table_name = table_name
        self.db_name = db_name
        # database has to work before creating tables
        # create_db(db_name=db_name)

    def create_table_full(self) -> bool:
        """
        create a table in the database
        note that in create session, the database is selected
        """
        session, schema = create_session(self.db_name)
        if not session:
            return False

        create_table_cmd = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            paper_title VARCHAR(255) NOT NULL UNIQUE,
            paper_arxiv_id VARCHAR(255) DEFAULT NULL UNIQUE,
            paper_arxiv_url VARCHAR(255) DEFAULT NULL UNIQUE,
            paper_pwc_url VARCHAR(255) DEFAULT NULL UNIQUE,
            github_url VARCHAR(255) DEFAULT NULL UNIQUE,

            contributors VARCHAR(255) DEFAULT NULL,
            build_sys_type VARCHAR(255) DEFAULT NULL,
            deps_file_url VARCHAR(255) DEFAULT NULL UNIQUE,
            deps_file_content_orig MEDIUMTEXT,

            build_status_orig VARCHAR(255) DEFAULT NULL,
            deps_file_content_edited MEDIUMTEXT,
            build_status_edited VARCHAR(255) DEFAULT NULL,
            datetime_latest_build DATETIME DEFAULT NULL,
            num_build_attempts INT DEFAULT 0,
            py_valid_versions VARCHAR(255) DEFAULT NULL,

            github_fork_url VARCHAR(255) DEFAULT NULL UNIQUE,
            pushed_to_fork BOOLEAN DEFAULT FALSE,
            pull_request_made BOOLEAN DEFAULT FALSE,
            tweet_posted BOOLEAN DEFAULT FALSE,
            tweet_url VARCHAR(255) DEFAULT NULL UNIQUE
        );"""

        try:
            # Check if the table exists
            table_exists = False
            tables = schema.get_tables()
            for table in tables:
                if table.get_name() == self.table_name:
                    table_exists = True
                    break

            if table_exists:
                print(f"Table {self.table_name} already exists.")
                return True

            # Create the table if it does not exist
            session.sql(create_table_cmd).execute()
            print(f"Table {self.table_name} created successfully.")
            return True
        except Exception as e:
            print(f"Error creating table: {str(e)}")
            return False
        finally:
            session.close()

    def populate_table_first_five_cols(self, row_limit: int = None) -> bool:
        """
        populate the table with data from data/links-between-papers-and-code.json
        sample:
        {
            "paper_url": "https://paperswithcode.com/paper/attngan-fine-grained-text-to-image-generation",
            "paper_title": "AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks",
            "paper_arxiv_id": "1711.10485",
            "paper_url_abs": "http://arxiv.org/abs/1711.10485v1",
            "paper_url_pdf": "http://arxiv.org/pdf/1711.10485v1.pdf",
            "repo_url": "https://github.com/bprabhakar/text-to-image",
            "is_official": false,
            "mentioned_in_paper": false,
            "mentioned_in_github": false,
            "framework": "pytorch"
        },

        Not all rows are inserted due to unique and not null constraints
        clashes mainly on paper_url, paper_arxiv_id
        Rows inserted: 185013 of 272525
        """
        session, schema = create_session(self.db_name)
        if not session: return False

        """ fill these fields
        paper_title VARCHAR(255) NOT NULL UNIQUE,
        paper_arxiv_id VARCHAR(255) DEFAULT NULL UNIQUE,
        paper_arxiv_url VARCHAR(255) DEFAULT NULL UNIQUE,
        paper_pwc_url VARCHAR(255) NOT NULL UNIQUE,
        github_url VARCHAR(255) NOT NULL,
        """

        rows_inserted = 0
        file_loc = os.path.join(ROOT, "data", "links-between-papers-and-code.json")
        data = None
        with open(file_loc, 'r', encoding='ascii') as f:
            data = json.load(f)

        table = schema.get_table(self.table_name)
        # populate only 5 columns
        try:
            rows_inserted = 0
            rows_skipped = 0

            def escape_value(value):
                if value is None:
                    return 'NULL'
                # Escape single quotes and backslashes
                return "'" + str(value).replace("'", "''").replace("\\", "\\\\") + "'"

            for idx, row in enumerate(data):
                if row_limit and idx >= row_limit:
                    break

                # Convert string to dict if needed
                if isinstance(row, str):
                    try:
                        row = json.loads(row)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing row #{idx}: {str(e)}")
                        continue

                # Skip if required unique field is None/NULL
                if not row.get('paper_url_abs'):
                    print(f"Skipping row #{idx}: Missing required paper_pwc_url")
                    rows_skipped += 1
                    continue

                # 'paper_title' : 'paper_title',
                # 'paper_arxiv_id' : 'paper_arxiv_id',
                # 'paper_arxiv_url' : 'paper_url_abs',
                # 'paper_pwc_url' : 'paper_url',
                # 'github_url' : 'repo_url'

                # Format values with proper escaping
                values = [
                    escape_value(row.get('paper_title')),
                    escape_value(row.get('paper_arxiv_id')),
                    escape_value(row.get('paper_url_abs')),
                    escape_value(row.get('paper_url')),
                    escape_value(row.get('repo_url')),
                ]

                insert_update_cmd = f"""
                INSERT INTO {self.table_name} (
                    paper_title, paper_arxiv_id, paper_arxiv_url, paper_pwc_url, github_url
                ) VALUES (
                    {values[0]}, {values[1]}, {values[2]}, {values[3]}, {values[4]}
                )
                """

                try:
                    session.sql(insert_update_cmd).execute()
                    session.commit()
                    rows_inserted += 1
                except Exception as e:
                    print(f"Error inserting row #{idx}: {str(e)}")
                    rows_skipped += 1
                    continue

            if row_limit:
                print(f"Rows inserted: {rows_inserted}, Rows skipped: {rows_skipped} of attempted {row_limit}")
            else:
                print(f"Rows inserted: {rows_inserted}, Rows skipped: {rows_skipped} of attempted {len(data)}")
            print(f"Total rows in table: {table.count()}")
        except Exception as e:
            print(f"Error populating table: {str(e)}")
            return False
        finally:
            session.close()
        return True

    def populate_table_additional_info(self) -> bool:
        """
        Populate additional columns in the table using GitHub repository data.
        This includes build_sys_type, deps_file_url, deps_file_content_orig, and contributors.
        """
        session, schema = create_session(self.db_name)
        if not session:
            return False

        rows_updated = 0
        table = schema.get_table(self.table_name)

        def escape_value(value):
            if value is None:
                return 'NULL'
            # Escape single quotes and backslashes
            return "'" + str(value).replace("'", "''").replace("\\", "\\\\") + "'"

        try:
            # Fetch all rows from the table
            rows = table.select('github_url, paper_title').execute().fetch_all()

            for row in rows:
                github_url = row[0]
                paper_title = row[1]

                if github_url:
                    # Extract owner and repo from the GitHub URL
                    owner_repo = self.extract_owner_repo(github_url)
                    if owner_repo:
                        owner, repo = owner_repo

                        # Try different common branch paths
                        possible_paths = [
                            f"https://github.com/{owner}/{repo}/blob/main/requirements.txt",
                            f"https://github.com/{owner}/{repo}/blob/master/requirements.txt",
                            f"https://raw.githubusercontent.com/{owner}/{repo}/main/requirements.txt",
                            f"https://raw.githubusercontent.com/{owner}/{repo}/master/requirements.txt"
                        ]

                        deps_file_content_orig = None
                        deps_file_url = None

                        for path in possible_paths:
                            content = self.get_file_content(path)
                            if content:
                                deps_file_content_orig = content
                                deps_file_url = path
                                break

                        # Set build_sys_type based on the presence of the requirements file
                        build_sys_type = 'requirements.txt' if deps_file_content_orig else None

                    # Get contributors
                    contributors = self.get_contributors(owner, repo)
                    if contributors and len(contributors) > 255:
                        contributors = contributors[:252] + '...'

                    # Escape all values for SQL
                    escaped_values = {
                        'build_sys_type': escape_value(build_sys_type),
                        'deps_file_url': escape_value(deps_file_url if deps_file_content_orig else None),
                        'deps_file_content_orig': escape_value(deps_file_content_orig),
                        'contributors': escape_value(contributors),
                        'paper_title': escape_value(paper_title)
                    }

                    # Update the table with the new information
                    update_cmd = f"""
                    UPDATE {self.table_name}
                    SET build_sys_type = {escaped_values['build_sys_type']},
                        deps_file_url = {escaped_values['deps_file_url']},
                        deps_file_content_orig = {escaped_values['deps_file_content_orig']},
                        contributors = {escaped_values['contributors']}
                    WHERE paper_title = {escaped_values['paper_title']};
                    """

                    try:
                        session.sql(update_cmd).execute()
                        rows_updated += 1
                        if rows_updated % 10 == 0:  # Progress update every 10 rows
                            print(f"Updated {rows_updated} rows...")
                    except Exception as e:
                        print(f"Error updating row for {paper_title}: {str(e)}")
                        continue

            print(f"Total rows updated with additional info: {rows_updated}")
            return True

        except Exception as e:
            print(f"Error populating additional info: {str(e)}")
            return False
        finally:
            session.close()

    def extract_owner_repo(self, github_url):
        """Extract owner and repository name from the GitHub URL."""
        regex = r"github\.com\/([^\/]+)\/([^\/]+)"
        match = re.search(regex, github_url)
        if match:
            return match.groups()
        return None

    def get_file_content(self, file_url):
        """Fetch the content of a file from the given URL."""
        try:
            # If the URL is a GitHub blob URL, convert it to raw format
            if 'github.com' in file_url and '/blob/' in file_url:
                file_url = file_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')

            response = requests.get(file_url)
            if response.status_code == 200:
                return response.text
            else:
                # Try to find requirements.txt in the repository
                if 'raw.githubusercontent.com' in file_url:
                    # Extract the repo path
                    parts = file_url.split('raw.githubusercontent.com/')[1].split('/')
                    owner, repo = parts[0], parts[1]

                    # List of common requirements file names and branches
                    req_files = ['requirements.txt']
                    branches = ['main', 'master'] # , 'dev', 'development']

                    # Try different combinations
                    for branch in branches:
                        for req_file in req_files:
                            try_url = f"https://raw.githubusercontent.com/{owner}/{repo}/blob/{branch}/{req_file}"
                            try:
                                resp = requests.get(try_url)
                                if resp.status_code == 200:
                                    return resp.text
                            except:
                                continue

                # print(f"Error fetching file content from {file_url}: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching file content: {str(e)}")
            return None

    def get_contributors(self, owner, repo):
        """Fetch contributors from the GitHub repository."""
        contributors_url = f"https://api.github.com/repos/{owner}/{repo}/contributors"

        # Get GitHub token from environment variable
        try:
            headers = {}
            if github_token:
                headers['Authorization'] = f'token {github_token}'
            else:
                print(f"Warning: No GitHub token found. Rate limits will be strict.")

            response = requests.get(contributors_url, headers=headers)

            # Check rate limits from response headers
            rate_limit = response.headers.get('X-RateLimit-Remaining', 'N/A')
            rate_reset = response.headers.get('X-RateLimit-Reset', 'N/A')
            if rate_reset != 'N/A':
                reset_time = datetime.fromtimestamp(int(rate_reset)).strftime('%Y-%m-%d %H:%M:%S')
            else:
                reset_time = 'N/A'

            if response.status_code == 403:
                if rate_limit == '0':
                    print(f"\nGitHub API rate limit exceeded!")
                    print(f"Rate limit will reset at: {reset_time}")
                else:
                    print(f"\nRepository {owner}/{repo} access forbidden (403)")
                    print(f"This might be a private repository or the token might not have sufficient permissions")
                return None

            elif response.status_code == 404:
                print(f"\nRepository {owner}/{repo} not found (404)")
                print(f"The repository might have been deleted or renamed")
                return None

            elif response.status_code == 200:
                contributors = [contributor['login'] for contributor in response.json()]
                if not contributors:
                    print(f"\nNo contributors found for {owner}/{repo}")
                    return None
                return ', '.join(contributors)
            else:
                print(f"\nError fetching contributors for {owner}/{repo}")
                print(f"Status code: {response.status_code}")
                print(f"Remaining API calls: {rate_limit}")
                print(f"Rate limit resets at: {reset_time}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"\nNetwork error fetching contributors for {owner}/{repo}")
            print(f"Error: {str(e)}")
            return None
        except Exception as e:
            print(f"\nUnexpected error fetching contributors for {owner}/{repo}")
            print(f"Error: {str(e)}")
            return None





    def populate_table_paper_repo_info_reqs(self) -> bool:
        r"""
        populate the table with data from data/paper_repo_info+reqs.json

        sample:
        ```
        {
            "paper_title": "AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks",
            "paper_url": "https://paperswithcode.com/paper/attngan-fine-grained-text-to-image-generation",
            "paper_arxiv_id": "1711.10485",
            "repo_url": "https://github.com/bprabhakar/text-to-image",
            "is_official": "false",
            "framework": "pytorch",
            "readmeUrl": "https://raw.githubusercontent.com/bprabhakar/text-to-image/master/README.md",
            "requirementsUrl": "https://raw.githubusercontent.com/bprabhakar/text-to-image/master/requirements.txt",
            "requirementsLastCommitDate": "2018-05-30T01:01:19Z",
            "mostProminentLanguage": "Jupyter Notebook",
            "stars": "17",
            "lastCommitDate": "2018-05-30T01:28:48Z",
            "contributors": "https://github.com/bprabhakar",
            "requirements": "Flask\npython-dateutil\neasydict\nscikit-image\nazure-storage-blob\napplicationinsights\nlibmc"
        }
        ```
        """

        session, schema = create_session(self.db_name)
        if not session: return False

        # Define the mapping between JSON keys (camelCase) and database columns (snake_case)
        key_mapping = {
            "paper_title": "paper_title",
            "paper_url": "paper_url",
            "paper_arxiv_id": "paper_arxiv_id",
            "repo_url": "repo_url",
            "is_official": "is_official",
            "framework": "framework",
            "readmeUrl": "readme_url",
            "requirementsUrl": "requirements_url",
            "requirementsLastCommitDate": "requirements_last_commit_date",
            "mostProminentLanguage": "most_prominent_language",
            "stars": "stars",
            "lastCommitDate": "last_commit_date",
            "contributors": "contributors",
            "requirements": "requirements"
        }

        columns = [
            "paper_title",
            "paper_url",
            "paper_arxiv_id",
            "repo_url",
            "is_official",
            "framework",
            "readme_url",
            "requirements_url",
            "requirements_last_commit_date",
            "most_prominent_language",
            "stars",
            "last_commit_date",
            "contributors",
            "requirements"
        ]

        # Function to convert JSON keys to database column names
        def convert_keys(row, key_mapping):
            converted_row = {}
            for json_key, db_key in key_mapping.items():
                if json_key in row:
                    converted_row[db_key] = row[json_key]
                else:
                    converted_row[db_key] = None
            return converted_row


        rows_inserted = 0
        file_loc = os.path.join(ROOT, "data", "paper_repo_info+reqs.json")
        data = None
        # with open(file_loc, 'r', encoding='utf-8') as f:
        with open(file_loc, 'r', encoding='utf-8') as f:
            data = json.load(f)

        table = schema.get_table(self.table_name)

        # there may be empty columns
        try:
            for idx, row in enumerate(data):
                # Convert JSON keys to database column names
                field_dict = convert_keys(row, key_mapping)
                # prepare data to conform to the schema
                row_values = []

                for col in columns:
                    if col.lower().endswith("date"):
                        try:
                            date_value = datetime.strptime(field_dict[col], "%Y-%m-%dT%H:%M:%SZ").date()
                            row_values.append(date_value.strftime("%Y-%m-%d"))  # Convert date to string
                        except (ValueError, TypeError):
                            row_values.append(None)
                    elif col == "is_official":
                        row_values.append(1 if field_dict[col] == "true" else 0)
                    elif col == "stars":
                        row_values.append(int(field_dict[col]) if field_dict[col] is not None else 0)
                    else:
                        row_values.append(field_dict[col])

                # Insert the row into the database
                try:
                    table.insert(columns).values(row_values).execute()
                    rows_inserted += 1
                except Exception as e:
                    print(f"Error inserting row #{idx}: {str(e)}")
                    continue

            print(f"Rows inserted: {rows_inserted} of attempted {len(data)}")
            print(f"Total rows: {table.count()}")

        except Exception as e:
            print(f"Error populating table: {str(e)}")
            # return False
        finally:
            session.close()
        return True

    def populate_table_papers_data(self) -> bool:
        """
        populate the table with data from data/papers_data.json
        in csv format so it can be read as a dataframe
        or converted to json

        sample:
        ```
        Title                Sapiens: Foundation for Human Vision Models
        Link           https://paperswithcode.com/paper/sapiens-found...
        Author Info            facebookresearch/sapiens •  • 22 Aug 2024
        Abstract       We present Sapiens, a family of models for fou...
        Tasks                        2D Pose Estimation|Depth Estimation
        GitHub Link          https://github.com/facebookresearch/sapiens
        Star Count                                                 3,686
        """
        session, schema = create_session(self.db_name)
        if not session: return False

        columns = [
            "title", "paper_url", "author_info", "abstract", "tasks", "github_url", "stars"
        ]

        rows_inserted = 0
        file_loc = os.path.join(ROOT, "data", "papers_data.csv")

        data = None
        with open(file_loc, 'r', encoding='utf-8') as f:
            data = pd.read_csv(f)

        table = schema.get_table(self.table_name)

        try:
            for idx, row in data.iterrows():
                row = row.to_dict()
                for k, v in row.items():
                    # convert NaN to None
                    if pd.isna(v):
                        row[k] = None
                    # strip leading/trailing whitespace if the value is a string
                    elif isinstance(v, str):
                        row[k] = v.strip()
                    # remove commas from star count and convert to int
                    if k == "Star Count":
                        if row[k] is None or row[k] == "None" or row[k] == "":
                            row[k] = 0
                        else:
                            row[k] = int(row[k].replace(",", ""))
                try:
                    # print(f'Inserting row #{idx}: {list(row.values())}')
                    table.insert(columns).values(list(row.values())).execute()
                    rows_inserted += 1
                except Exception as e:
                    print(f"Error inserting row #{idx}: {str(e)}")
                    continue

            print(f"Rows inserted: {rows_inserted} of attempted {len(data)}")
            print(f"Total rows in table: {table.count()}")

        except Exception as e:
            print(f"Error populating table: {str(e)}")
            return False
        finally:
            session.close()
        return True

    def populate_table_build_check_results(self) -> bool:
        """
        populate the table with data from output/build_check_results.csv
        only has 'file_or_repo', 'status' fields
        sample:
        github_link	updated_requirements
        https://github.com/Maymaher/StackGANv2
        "torch==1.0.0 torchvision==0.2.1 numpy==1.15.1 lmdb==0.94
            easydict==1.9 six==1.11.0 requests==2.19.1 pandas==0.23.4
            Pillow==5.4.1 python_dateutil==2.7.5 tensorboardX==1.6
            PyYAML==3.13"
        """
        session, schema = create_session(self.db_name)
        if not session: return False

        columns = ["file_or_repo", "status"]

        rows_total = 0
        file_loc = os.path.join(ROOT, "output", "build_check_results.csv")

        data = None
        with open(file_loc, 'r', encoding='utf-8') as f:
            # data = json.load(f)
            data = pd.read_csv(f)

        table = schema.get_table(self.table_name)

        try:
            for idx, row in data.iterrows():
                row = row.to_dict()
                file_or_repo, status = None, 'open'

                for k, v in row.items():
                    if k == "file_or_repo":
                        file_or_repo = v.strip()
                    elif k == "status":
                        status = v.strip()

                try:
                    table.insert(columns).values([file_or_repo, status]).execute()
                    rows_total += 1
                except Exception as e:
                    print(f"Error inserting row #{idx}: {str(e)}")
                    continue

            print(f"Rows inserted: {rows_total} of attempted {len(data)}")
            print(f"Total rows: {table.count()}")

        except Exception as e:
            print(f"Error populating table: {str(e)}")
            return False
        finally:
            session.close()
        return True

    def populate_table_issues_classified(self) -> bool:
        """
        populate the table with data from output/issues_classified.csv
        only has 'title', 'body' fields
        """
        session, schema = create_session(self.db_name)
        if not session: return False

        # columns = ["title", "body", "labels", "comments_count", "state", "is_version_issue"]

        columns = ["title", "body"]

        rows_inserted = 0
        file_loc = os.path.join(ROOT, "output", "issues_classified.csv")
        data = None
        with open(file_loc, 'r', encoding='utf-8') as f:
            data = pd.read_csv(f)

        table = schema.get_table(self.table_name)

        try:
            for idx, row in data.iterrows():
                row = row.to_dict()
                title, body = None, None
                try:
                    title = row["title"]
                except KeyError:
                    pass
                try:
                    body = row["body"]
                except KeyError:
                    pass

                try:
                    table.insert(columns).values([title, body]).execute()
                    rows_inserted += 1
                except Exception as e:
                    print(f"Error inserting row #{idx}: {str(e)}")
                    continue

            print(f"Rows inserted: {rows_inserted} of attempted {len(data)}")
            print(f"Total rows: {table.count()}")

        except Exception as e:
            print(f"Error populating table: {str(e)}")
            return False
        finally:
            session.close()
        return True

    def populate_table_updated_requirements(self) -> bool:
        """
        populate the table with data from output/updated_requirements_results.csv
        only has 'github_link', 'updated_requirements' fields
        """
        session, schema = create_session(self.db_name)
        if not session: return False

        columns = ["github_url", "updated_requirements"]

        rows_inserted = 0
        file_loc = os.path.join(ROOT, "output", "updated_requirements_results.csv")
        data = None
        with open(file_loc, 'r', encoding='ascii') as f:
            data = pd.read_csv(f)

        table = schema.get_table(self.table_name)

        for idx, row in enumerate(data):
            if row_limit and idx >= row_limit:
                break

            row_values = [
                row.get('paper_title'),
                row.get('paper_arxiv_id'),
                row.get('paper_arxiv_url'),
                row.get('paper_pwc_url'),
                row.get('github_url')
            ]

            insert_update_cmd = f"""
                INSERT INTO {self.table_name} (
                    paper_title, paper_arxiv_id, paper_arxiv_url, paper_pwc_url, github_url
                ) VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    paper_arxiv_id = VALUES(paper_arxiv_id),
                    paper_arxiv_url = VALUES(paper_arxiv_url),
                    paper_pwc_url = VALUES(paper_pwc_url),
                    github_url = VALUES(github_url);
            """

            try:
                session.sql(insert_update_cmd).execute(row_values)
                session.commit()
                rows_inserted += 1
            except Exception as e:
                print(f"Error inserting row #{idx}: {str(e)}")
                continue

            print(f"Rows inserted: {rows_inserted} of attempted {len(data)}")
            print(f"Total rows: {table.count()}")

        # except Exception as e:
        #     print(f"Error populating table: {str(e)}")
        #     return False
        # finally:
        #     session.close()
        # return True


if __name__ == '__main__':
    database_name = "grimrepor_database"

    spinup_mysql_server()
    show_databases()
    create_db(db_name=database_name) # do once
    show_all_tables(db_name=database_name)
    # added user confirmation to drop all tables sa this is a destructive operation
    drop_all_tables(db_name=database_name)

    papers_and_code = Table(table_name="papers_and_code", db_name=database_name)
    papers_and_code.create_table_full()
    show_table_columns("papers_and_code", db_name=database_name)
    papers_and_code.populate_table_first_five_cols(row_limit=10)
    show_table_contents("papers_and_code", db_name=database_name, limit_num=10)

    papers_and_code.populate_table_additional_info()
    show_table_contents("papers_and_code", db_name=database_name, limit_num=10)

    # have function to populate the table from each data source
    # FIND      first 5 cols from 'links-between-papers-and-code.json'
    # QUALIFY   3 columns
    # BUILD
    # FIX
    # PUBLISH  git fork, tweet
    # # import function into another file to do the cell updates

    # TODO: optimize speed
    # DESIGN CHOICE: there are some papers with 10+ repos for example,
    # so consider if we want to link duplicates, have a separate table for duplicates,
    # keep track of which paper title corresponds and then delete all entries related after bulk upload

    show_all_tables(db_name=database_name)


# use a .env file at the root of the project with the following:
# MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD
