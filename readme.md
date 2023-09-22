This step-by-step list should help simplify the process of setting up and running the program.


#CREATE A NEW ENVIRONMENT
python -m venv myenv


CHANGE TO NEW ENVIRONMENT
FOR WINDOWS BASH
source myenv/bin/activate

FOR LINUX AND MAC BASH
source myenv/bin/activate

Thereafter:

Create a Twitter Developer Account:

    Sign up for a Twitter Developer account to obtain API access keys. The type of keys you obtain will influence the quality and quantity of data you can scrape.

Create an OpenAI Account:

    Sign up for an OpenAI account. Once registered, you will receive OpenAI keys which can be retrieved from the settings page of your account.

Setup Development Environment:

    It's recommended to use Visual Studio Code (VS Code) as your Integrated Development Environment (IDE).
    If VS Code cannot be installed, you can use the web version provided by Gitpod, which may simplify and speed up the implementation process.

Retrieve and Store Access Keys:

    Once you have obtained the Twitter and OpenAI keys, copy and paste them into the phase1 and phase2 files respectively, by serching api using ctrl F.

Install Required Packages:

    Execute the requirements.txt file to install all necessary packages for the program. If these packages are not pre-installed on your computer, you can install them all at once by running the following command in your terminal:

        pip install -r requirements.txt

    Run the Program:
        Open two separate terminal windows.
        In one terminal, run the phase1.py file.
        In the other terminal, run the phase2.py file. Ensure that phase2.py is active, as phase1.py will communicate with it post the data scraping process.

    Follow On-screen Prompts:
        Proceed with the on-screen instructions or requests facilitated through user inputs to navigate through the subsequent steps of the program.





