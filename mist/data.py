import numpy as np
import random

def get_data(which_mist):
    if which_mist == "MIST-20":
        mist_items = [
             "Government Officials Have Manipulated Stock Prices to Hide Scandals",
             "The Corporate Media Is Controlled by the Military-Industrial Complex: The Major Oil Companies Own the Media and Control Their Agenda",
             "New Study: Left-Wingers Are More Likely to Lie to Get a Higher Salary",
             "The Government Is Manipulating the Public's Perception of Genetic Engineering in Order to Make People More Accepting of Such Techniques",
             "Left-Wing Extremism Causes 'More Damage' to World Than Terrorism, Says UN Report",
             "Certain Vaccines Are Loaded with Dangerous Chemicals and Toxins",
             "New Study: Clear Relationship Between Eye Color and Intelligence",
             "The Government Is Knowingly Spreading Disease Through the Airwaves and Food Supply",
             "Ebola Virus 'Caused by US Nuclear Weapons Testing', New Study Says",
             "Government Officials Have Illegally Manipulated the Weather to Cause Devastating Storms",
             "Attitudes Toward EU Are Largely Positive, Both Within Europe and Outside It",
             "One-in-Three Worldwide Lack Confidence in Non-Governmental Organizations",
             "Reflecting a Demographic Shift, 109 US Counties Have Become Majority Nonwhite Since 2000",
             "International Relations Experts and US Public Agree: America Is Less Respected Globally",
             "Hyatt Will Remove Small Bottles from Hotel Bathrooms",
             "Morocco’s King Appoints Committee Chief to Fight Poverty and Inequality",
             "Republicans Divided in Views of Trump’s Conduct, Democrats Are Broadly Critical",
             "Democrats More Supportive than Republicans of Federal Spending for Scientific Research",
             "Global Warming Age Gap: Younger Americans Most Worried",
             "US Support for Legal Marijuana Steady in Past Year"
            ]
        labels = ["Fake","Fake","Fake","Fake","Fake","Fake","Fake","Fake","Fake","Fake",
                                       "Real","Real","Real","Real","Real","Real","Real","Real","Real","Real"]
    if which_mist == "MIST-16":
        mist_items = [
            "The Government Is Knowingly Spreading Disease Through the Airwaves and Food Supply",
            "The Government Is Actively Destroying Evidence Related to the JFK Assassination",
            "Government Officials Have Manipulated Stock Prices to Hide Scandals",
            "A Small Group of People Control the World Economy by Manipulating the Price of Gold and Oil",
            "The Government Is Conducting a Massive Cover-Up of Their Involvement in 9/11",
            "New Study: Left-Wingers Are More Likely to Lie to Get a Higher Salary",
            "Climate Scientists' Work Is 'Unreliable', a 'Deceptive Method of Communication''",
            "Left-Wingers Are More Likely to Lie to Get a Good Grade",
             "Morocco’s King Appoints Committee Chief to Fight Poverty and Inequality",
             "US Hispanic Population Reached New High in 2018, But Growth Has Slowed",
             "Hyatt Will Remove Small Bottles from Hotel Bathrooms",
             "Taiwan Seeks to Join Fight Against Global Warming",
             "About a Quarter of Large US Newspapers Laid off Staff in 2018",
             "Majority in US Still Want Abortion Legal, with Limits",
             "Most Americans Say It’s OK for Professional Athletes to Speak out Publicly about Politics",
             "United Nations Gets Mostly Positive Marks from People Around the World"
            ]
        labels = ["Fake","Fake","Fake","Fake","Fake","Fake","Fake","Fake",
                                       "Real","Real","Real","Real","Real","Real","Real","Real"]
    
    combined = list(zip(mist_items, labels))
    random.shuffle(combined)
    mist_items_shuffled, labels_shuffled = zip(*combined)
    mist_items_shuffled = list(mist_items_shuffled)
    labels_shuffled = list(labels_shuffled)
    return mist_items_shuffled, labels_shuffled

def get_prompt(user_description, items):
    item_description = ""
    for i in range(len(items)):
        item_description += f"news_{i+1}: {items[i]}\n"
    return f"""{user_description}\n# Please categorize the following news headlines as either
'Fake News' or 'Real News'.\n{item_description}\nSome items may look credible or obviously false at first 
sight, but may actually fall in the opposite category. However, for each news headline, only one
category is correct.\n
*Make sure you answered every question.*
Your answer should closely follow the response format:
{{
    "news_1": "Fake",
    "news_2": "Real",
    ...
}}
Ensure that your output can be directly converted into **JSON format**, and avoid outputting anything unnecessary!
"""
